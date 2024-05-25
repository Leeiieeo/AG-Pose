import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from utils.solver_housecat6d import test_func, get_logger
from provider.housecat6d_dataset import HouseCat6DTestDataset
from utils.housecat6d_eval_utils import evaluate_housecat
from Net import Net


def get_parser():
    parser = argparse.ArgumentParser(
        description="evaluate on housecat6d dataset")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/HouseCat6D/housecat6d.yaml",
                        help="path to config file")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=150,
                        help="test epoch")
    parser.add_argument("--result_dir",
                        type=str,
                        default="results",
                        help="")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    
    exp_name = args.config.split("/")[-1].split(".")[0]
    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.log_dir = os.path.join('log', exp_name)
    cfg.save_path = os.path.join(cfg.log_dir, args.result_dir)
    if not os.path.isdir(cfg.save_path):
        os.makedirs(cfg.save_path)
        
    cfg.ckpt_dir = os.path.join(cfg.log_dir, 'ckpt')

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+f"/test_logger_{cfg.test_epoch}.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> loading model ...")
    model = Net(cfg.pose_net)
    model = model.cuda()
    checkpoint = os.path.join(cfg.ckpt_dir, 'epoch_' + str(cfg.test_epoch) + '.pt')
    logger.info("=> loading checkpoint from: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    # data loader
    dataset = HouseCat6DTestDataset(cfg.test_dataset.img_size, cfg.test_dataset.sample_num, cfg.test_dataset.dataset_dir)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            shuffle=False,
            drop_last=False
        )
    test_func(model, dataloder, cfg.save_path)
    evaluate_housecat(cfg.save_path, logger)