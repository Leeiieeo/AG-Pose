import os
import sys
import argparse
import random
import math

import torch
import gorilla
import _pickle as cPickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


from Net import Net
from nocs_dataset import TestDataset

from draw_utils import *

sym_ids = [1, 2, 4]

def eliminate_y_rotation(rotation):
    theta_x = rotation[0, 0] + rotation[2, 2]
    theta_y = rotation[0, 2] - rotation[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                        [0.0,            1.0,  0.0           ],
                        [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = rotation @ s_map
    
    return rotation

def get_result(end_points):
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
    
    return result


def get_parser():
    parser = argparse.ArgumentParser(
        description="Visualize Pose Estimation Results")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/camera_real.yaml",
                        help="path to config file")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=30,
                        help="test epoch")
    parser.add_argument("--draw_dir",
                        type=str,
                        default="./viz_results",
                        help="draw_dir")
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.ckpt_dir = os.path.join(cfg.log_dir, 'ckpt')
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.draw_dir = args.draw_dir
    os.makedirs(cfg.draw_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.draw_dir, "ours"), exist_ok=True)
    os.makedirs(os.path.join(cfg.draw_dir, "ours_err"), exist_ok=True)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    return cfg

if __name__ == "__main__":
    cfg = init()

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    draw_dir = cfg.draw_dir

    # model
    ours_model = Net(cfg.pose_net).cuda()
    
    checkpoint = os.path.join(cfg.ckpt_dir, 'epoch_' + str(cfg.test_epoch) + '.pt')
    print("=> loading checkpoint from path: {} ...".format(checkpoint))
    gorilla.solver.load_checkpoint(model=ours_model, filename=checkpoint)
 
    ours_model.eval()
    # data loader
    dataset = TestDataset(cfg.test_dataset.img_size, cfg.test_dataset.sample_num, \
        cfg.test_dataset.dataset_dir, cfg.setting, cfg.test_dataset.dataset_name)
    
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            shuffle=False,
            drop_last=False
        )
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloder)):
            # path = data['path'][0]
            path = dataloder.dataset.result_pkl_list[i]
            inputs = {
                    'rgb': data['rgb'][0].cuda(),
                    'pts': data['pts'][0].cuda(),
                    'choose': data['choose'][0].cuda(),
                    'category_label': data['category_label'][0].cuda(),
                }
            # ['pred_translation', 'pred_rotation', 'pred_size', 'pred_kpt_3d', 'kpt_nocs']
            end_points_ours = ours_model(inputs) 
 
            ours_result = get_result(end_points=end_points_ours)
            
            with open(path, 'rb') as f:
                data = cPickle.load(f)
        
            image_path = data['image_path']
            image = cv2.imread(image_path + '_color.png')[:, :, :3]
            image = image[:, :, ::-1] #480*640*3 RGB
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
            
            scene_id, image_id = image_path.split('/')[-2:]

            ins_num = inputs["rgb"].shape[0]
            translation_gt = np.zeros((ins_num, 3), dtype=np.float32)
            rotation_gt = np.zeros((ins_num, 3, 3), dtype=np.float32)
            scale_gt = np.zeros((ins_num), dtype=np.float32)
            size_gt = ours_result['gt_scales']
            

            for idx in range(ins_num):
                if idx >= ours_result['gt_RTs'].shape[0]:
                    continue
                cat_id = ours_result["gt_class_ids"][idx]
                s = np.cbrt(np.linalg.det(ours_result['gt_RTs'][idx][:3, :3]))
                R = ours_result['gt_RTs'][idx][:3, :3] / s
                if (cat_id in sym_ids) or ((cat_id == 6) and ours_result["gt_handle_visibility"][idx] ==0):
                    R = eliminate_y_rotation(R)               
                scale_gt[idx] = s
                T = ours_result['gt_RTs'][idx][:3, 3]
                translation_gt[idx] = T
                rotation_gt[idx] = R
            gt_sR = rotation_gt / scale_gt[:, np.newaxis, np.newaxis]

            draw_image_bbox_ours = image.copy()    
            draw_image_ours_kpt = image.copy()  
            draw_nocs_err_ours = image.copy()    

            ours_path = os.path.join(draw_dir, "ours", '{}_{}_bbox.png'.format(scene_id, image_id))
            ours_err_path = os.path.join(draw_dir, "ours_err", '{}_{}_bbox.png'.format(scene_id, image_id))

            for idx, RTs in enumerate(ours_result['gt_RTs']):
                
                cat_id = ours_result["gt_class_ids"][idx]
                if (cat_id in sym_ids) or ((cat_id == 6) and (ours_result["gt_handle_visibility"][idx] ==0)):
                    s = np.cbrt(np.linalg.det(RTs[:3, :3]))
                    R = RTs[:3, :3] / s
                    R = eliminate_y_rotation(R)
                    R = R * s
                    RTs[:3, :3] = R
                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RTs)
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)
                
                bbox_3d = get_3d_bbox(ours_result['gt_scales'][idx], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RTs)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox_ours = draw(draw_image_bbox_ours, projected_bbox, projected_axes, (0, 255, 0))
            
            num_pred_instances = len(ours_result['pred_class_ids'])

            for idx in range(num_pred_instances):
                class_label = ours_result["pred_class_ids"][idx]
                gt_idx = np.where(ours_result["gt_class_ids"] == class_label)[0]
                if gt_idx.size != 1:
                    continue
                gt_idx = gt_idx[0]
                if gt_idx >= translation_gt.shape[0]:
                    continue
                
                pts = inputs['pts'][idx].detach().cpu().numpy()
                
                th = 0.2
                pts_num = pts.shape[0]
                pts_2d = calculate_2d_projections(pts.transpose(), intrinsics)
                     
                pred_kpt_3d = end_points_ours['pred_kpt_3d'][idx].detach().cpu().numpy()
                pred_kpt_nocs = end_points_ours['kpt_nocs'][idx].detach().cpu().numpy()
                kpt_nocs_gt = (pred_kpt_3d - translation_gt[gt_idx]) @ gt_sR[gt_idx].reshape(3, 3) # (1024, 3)
                ours_kpt_nocs_err = np.linalg.norm(kpt_nocs_gt - pred_kpt_nocs, axis=1)
                ours_kpt_nocs_err = np.clip(ours_kpt_nocs_err, 0, th)
                ours_kpt_nocs_err = (ours_kpt_nocs_err) / (th)
                ours_err_colors = ours_kpt_nocs_err[:, None] * (RED - GREEN)[None, :] + GREEN[None, :]
                ours_err_colors = ours_err_colors * 255
                kpt_num = pred_kpt_3d.shape[0]
                kpt_2d = calculate_2d_projections(pred_kpt_3d.transpose(), intrinsics)
                for i in range(kpt_num):
                    x, y = kpt_2d[i]
                    color = ours_err_colors[i]
                    draw_image_ours_kpt = cv2.circle(draw_image_ours_kpt, (x, y), 2, GREEN*255, -1)
                    draw_nocs_err_ours = cv2.circle(draw_nocs_err_ours, (x, y), 3, tuple(color.tolist()), -1)  
            cv2.imwrite(ours_err_path, draw_nocs_err_ours[:, :, ::-1])    
            
            # draw_ours
            num_pred_instances = len(ours_result['pred_class_ids'])
        
            for idx in range(num_pred_instances): 
                class_label = ours_result["pred_class_ids"][idx]
                RTs = ours_result['pred_RTs'][idx]
                
                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RTs)
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

                bbox_3d = get_3d_bbox(ours_result['pred_scales'][idx], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RTs)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox_ours = draw(draw_image_bbox_ours, projected_bbox, projected_axes, (255, 0, 0))
                
            cv2.imwrite(ours_path, draw_image_bbox_ours[:, :, ::-1])
                



