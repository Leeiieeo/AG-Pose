import logging
import os
import pickle as cPickle
import time

import gorilla
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg, start_epoch=1, start_iter=0):
        super(Solver, self).__init__(
            model=model,
            dataloaders=dataloaders,
            cfg=cfg,
            logger=logger,
        )
        self.dataset_name = cfg.train_dataset.dataset_name
        self.loss = loss
        self.logger.propagate = 0
        tb_writer_ = tools_writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer = tb_writer_

        self.per_val = cfg.per_val
        self.per_write = cfg.per_write
        self.epoch = start_epoch
        self.iter = start_iter
        
        if start_epoch != 1:
            self.lr_scheduler.last_epoch = start_iter
        
    def solve(self):
        while self.epoch <= self.cfg.max_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value
            
            ckpt_path = os.path.join(
                self.cfg.ckpt_dir, 'epoch_' + str(self.epoch) + '.pt')
            torch.save(self.model.state_dict(), ckpt_path)
            
            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            write_info += f"lr: {self.lr_scheduler.get_lr()[0]:.5f}"
            self.logger.warning(write_info)
            self.epoch += 1

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()

        for k in self.dataloaders.keys():
            self.dataloaders[k].dataset.reset()
        i=0

        if self.dataset_name == "camera_real":
            data_iter = zip(self.dataloaders["syn"], self.dataloaders["real"])
            iter_lenth = len(self.dataloaders["syn"])
        elif self.dataset_name == "camera":
            data_iter = self.dataloaders["syn"]
            iter_lenth = len(self.dataloaders["syn"])
        else:
            raise NotImplementedError
        
        for train_data in data_iter:
            data_time = time.time()-end

            self.optimizer.zero_grad()
            method = getattr(self, f"step_{self.dataset_name}")
            loss, dict_info_step = method(train_data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()
            backward_time = time.time() - end - forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                self.log_buffer.average(self.per_write)
                prefix = '[{}/{}][{}/{}][{}] Train - '.format(
                    self.epoch, self.cfg.max_epoch, i, iter_lenth, self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.iter += 1
            i+=1

        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def evaluate(self):
        mode = 'eval'
        self.model.eval()

        for i, data in enumerate(self.dataloaders["eval"]):
            with torch.no_grad():
                _, dict_info_step = self.step(data, mode)
                self.log_buffer.update(dict_info_step)
                if i % self.per_write == 0:
                    self.log_buffer.average(self.per_write)
                    prefix = '[{}/{}][{}/{}] Test - '.format(
                        self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["eval"]))
                    write_info = self.get_logger_info(
                        prefix, dict_info=self.log_buffer._output)
                    self.logger.info(write_info)
                    self.write_summary(self.log_buffer._output, mode)
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def step_camera_real(self, train_data, mode):
        syn_data, real_data = train_data
        
        b1 = syn_data['rgb'].size(0)
        b2 = real_data['rgb'].size(0)

        for key in syn_data:
            syn_data[key] = syn_data[key].cuda()
        for key in real_data:
            real_data[key] = real_data[key].cuda()

        data = {
            'rgb': torch.cat([syn_data['rgb'], real_data['rgb']], dim=0),
            'pts': torch.cat([syn_data['pts'], real_data['pts']], dim=0),
            'choose': torch.cat([syn_data['choose'], real_data['choose']], dim=0),
            'category_label': torch.cat([syn_data['category_label'], real_data['category_label']], dim=0),
        }
        end_points = self.model(data)

        for key in end_points:
            syn_data[key] = end_points[key][0:b1]
            real_data[key] = end_points[key][b1:]
            
        loss_dict_syn = self.loss(syn_data)
        loss_dict_real = self.loss(real_data)
        
        dict_info = {}
        for k in loss_dict_syn.keys():
            dict_info[k] = (b1*float(loss_dict_syn[k].item()) + b2*float(loss_dict_real[k].item())) / (b1+b2)
            
        dict_info['loss_syn'] = float(loss_dict_syn['loss_all'].item())
        dict_info['loss_real'] = float(loss_dict_real['loss_all'].item())
        
        loss_all = (loss_dict_syn['loss_all']*b1 + loss_dict_real['loss_all']*b2) / (b1+b2)

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_lr()[0]

        return loss_all, dict_info
    
    def step_camera(self, train_data, mode):
        syn_data = train_data

        for key in syn_data:
            syn_data[key] = syn_data[key].cuda()
        
        end_points = self.model(syn_data)

        for key in end_points:
            syn_data[key] = end_points[key]
            
        loss_dict = self.loss(syn_data)
        
        dict_info = {}
        for k, v in loss_dict.items():
            dict_info[k] = float(v.item())
        
        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_lr()[0]

        return loss_dict['loss_all'], dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            else:
                info = info + '{}: {:.6f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False
    
def test_func(model, dataloder, save_path):
    model.eval()
    time_all = 0
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            path = dataloder.dataset.result_pkl_list[i]

            inputs = {
                'rgb': data['rgb'][0].cuda(),
                'pts': data['pts'][0].cuda(),
                'choose': data['choose'][0].cuda(),
                'category_label': data['category_label'][0].cuda(),
            }
            
            start_time = time.time()
            end_points = model(inputs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_all += elapsed_time

            pred_translation = end_points['pred_translation']
            pred_size = end_points['pred_size']
            pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
            pred_size = pred_size / pred_scale
            pred_rotation = end_points['pred_rotation']

            num_instance = pred_rotation.size(0)
            pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
            pred_RTs[:, :3, 3] = pred_translation
            pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
            pred_scales = pred_size

            # save
            result = {}

            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()

            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()

            result['gt_scales'] = data['gt_scales'][0].numpy()
            result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()

            result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            result['pred_scores'] = data['pred_scores'][0].numpy()

            result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
            result['pred_scales'] = pred_scales.detach().cpu().numpy()

            with open(os.path.join(save_path, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(result, f)

            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, len(dataloder), num_instance)
            )

            t.update(1)
    print(f"time_all: {time_all}")
            
class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        # writer = SummaryWriter(dir_project)
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger