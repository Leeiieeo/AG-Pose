import os
import math
import cv2
# import open3d as o3d
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
# from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.housecat6d_utils import (
    load_housecat_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,

)

class HouseCat6DTrainingDataset(Dataset):
    def __init__(self,
            image_size,
            sample_num,
            data_dir,       
            seq_length=-1, # -1 means full
            img_length=-1, # -1 means full
    ):
        self.sample_num = sample_num
        self.data_dir = data_dir
        self.img_size = image_size
        self.train_scenes_rgb = glob.glob(os.path.join(self.data_dir,'scene*','rgb'))
        self.train_scenes_rgb.sort()
        self.train_scenes_rgb = self.train_scenes_rgb[:seq_length] if seq_length != -1 else self.train_scenes_rgb[:]
        self.real_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.train_scenes_rgb]
        
        self.meta_list = [os.path.join(scene, '..', 'meta.txt') for scene in self.train_scenes_rgb]
        self.min_num = 100
        for meta in self.meta_list:
            with open(meta, 'r') as file:
                content = file.read()
                num_count = content.count('\n') + 1
            self.min_num = num_count if num_count < self.min_num else self.min_num
        
        self.real_img_list = []
        for scene in self.train_scenes_rgb:
            img_paths = glob.glob(os.path.join(scene, '*.png'))
            img_paths.sort()
            img_paths = img_paths[:img_length] if img_length != -1 else img_paths[:]
            for img_path in img_paths:
                self.real_img_list.append(img_path)

        print(f'{len(self.train_scenes_rgb)} sequences, {img_length} images per sequence. Total {len(self.real_img_list)} images are found.')

        self.xmap = np.array([[i for i in range(1096)] for j in range(852)]) # 1096*852
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.3)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.sym_ids = [1, 2, 7]
        self.reset()

    def __len__(self):
        return len(self.real_img_list)

        
    def reset(self):
        num_real_img = len(self.real_img_list)
        self.img_index = np.arange(num_real_img)
        np.random.shuffle(self.img_index)

    def __getitem__(self, index):
        image_index = self.img_index[index]
        data_dict = self._read_data(image_index)
        assert data_dict is not None
        return data_dict
        
    def _read_data(self, image_index):
        img_path = os.path.join(self.data_dir, self.real_img_list[image_index])
        real_intrinsics = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3,3)
        cam_fx, cam_fy, cam_cx, cam_cy = real_intrinsics[0,0], real_intrinsics[1,1], real_intrinsics[0,2], real_intrinsics[1,2]

        depth = load_housecat_depth(img_path)   # (852, 1096)
        depth = fill_missing(depth, self.norm_scale, 1)

        # mask
        with open(img_path.replace('rgb','labels').replace('.png','_label.pkl'), 'rb') as f:
            gts = cPickle.load(f)
        num_instance = len(gts['instance_ids'])
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path.replace('rgb','instance'))[:, :, 2] # TODO (852, 1096)

        idx = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], img_width=852, img_length=1096)
        mask = np.equal(mask, gts['instance_ids'][idx])
        mask = np.logical_and(mask , depth > 0)

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            return None
        elif len(choose) <= self.sample_num:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        pts2 = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1))[choose] / self.norm_scale
        pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32) # (npts, 3)
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        # rgb
        rgb = cv2.imread(img_path)[:, :, :3] # TODO (852, 1096, 3)
        rgb = rgb[:, :, ::-1] # (852, 1096, 3)
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
        rgb = np.array(rgb)
        rgb = rgb_add_noise(rgb)
        rgb = self.transform(rgb)

        # update choose
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        
        # gt
        translation = gts['translations'][idx].astype(np.float32)
        rotation = gts['rotations'][idx].astype(np.float32)
        size = gts['gt_scales'][idx].astype(np.float32)

        if cat_id in self.sym_ids:
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                [0.0,            1.0,  0.0           ],
                                [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map

        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        
        return ret_dict 
    
class HouseCat6DTestDataset():
    def __init__(self,
                 image_size, 
                 sample_num, 
                 data_dir
                ):
        self.data_dir = data_dir
        self.sample_num = sample_num
        self.img_size = image_size
        self.test_scenes_rgb = glob.glob(os.path.join(self.data_dir, 'test_scene*', 'rgb'))
        self.test_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.test_scenes_rgb]
        self.test_img_list = [img_path for scene in self.test_scenes_rgb for img_path in
                              glob.glob(os.path.join(scene, '*.png'))]

        n_image = len(self.test_img_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(1096)] for j in range(852)])
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.norm_scale = 1000.0    # normalization scale
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])


    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, index):
        img_path = self.test_img_list[index]
        with open(img_path.replace('rgb', 'labels').replace('.png', '_label.pkl'), 'rb') as f:
            gts = cPickle.load(f)

        # pred_mask = pred_data['pred_masks']
        mask_path = img_path.replace("rgb", "instance")
        mask = cv2.imread(mask_path)
        assert mask is not None
        mask = mask[:, :, 2]
        num_instance = len(gts['class_ids'])

        # rgb
        rgb = cv2.imread(img_path)[:, :, :3] # TODO 1096x852
        rgb = rgb[:, :, ::-1] # (852, 1096, 3)
        
        # pts
        intrinsics = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3, 3)
        cam_fx, cam_fy, cam_cx, cam_cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        depth = load_housecat_depth(img_path) #480*640 # TODO 1096x852
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # (852, 1096, 3)

        all_rgb = []
        all_pts = []
        all_choose = []
        all_cat_ids = []
        flag_instance = torch.zeros(num_instance) == 1
        mask_target = mask.copy().astype(np.float32)


        for j in range(num_instance):
            mask = np.equal(mask_target, gts['instance_ids'][j])
            inst_mask = 255 * mask.astype('uint8')
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            if np.sum(mask) > 16:
                rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][j], img_width=852, img_length=1096)
                # rmin, rmax, cmin, cmax = get_bbox_from_mask(mask, img_width = 852, img_length = 1096)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                cat_id = gts['class_ids'][j] - 1 # convert to 0-indexed
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :].copy()

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))
                
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1

        ret_dict = {}
        RTs = []
        s = np.linalg.norm(gts["gt_scales"], axis=1)
        for each_idx in range(len(gts["rotations"])):
            matrix = np.identity(4)
            matrix[:3, :3] = gts["rotations"][each_idx] * s[each_idx]
            matrix[:3, 3] = gts["translations"][each_idx]
            RTs.append(matrix)
        RTs = np.stack(RTs, 0)
        
        ret_dict['gt_class_ids'] = torch.tensor(gts['class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(gts["bboxes"])
        ret_dict['gt_RTs'] = torch.tensor(RTs)
        ret_dict['gt_scales'] = torch.tensor(gts["gt_scales"] / s[:, np.newaxis])
        ret_dict['index'] = index

        if len(all_pts) == 0:
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))
        else:
            ret_dict['choose'] = torch.stack(all_choose)
            ret_dict['pts'] = torch.stack(all_pts) # N*3
            ret_dict['rgb'] = torch.stack(all_rgb)
            ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])[flag_instance==1]
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])[flag_instance==1]
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))[flag_instance==1]

        return ret_dict
