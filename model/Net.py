import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses import ChamferDis, PoseDis, SmoothL1Dis, ChamferDis_wo_Batch
from utils.data_utils import generate_augmentation
from model.modules import ModifiedResnet, PointNet2MSG
from model.Net_modules import InstanceAdaptiveKeypointDetector, GeometricAwareFeatureAggregator, PoseSizeEstimator, NOCS_Predictor, Reconstructor

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cat_num = cfg.cat_num
        self.cfg = cfg
        if cfg.rgb_backbone == "resnet":
            self.rgb_extractor = ModifiedResnet()
        elif cfg.rgb_backbone == 'dino':
            # frozen dino
            self.rgb_extractor = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False

            self.feature_mlp = nn.Sequential(
                nn.Conv1d(384, 128, 1),
            )
        else:
            raise NotImplementedError
        
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        
        self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.GAFA = GeometricAwareFeatureAggregator(cfg.GAFA)

        self.nocs_predictor = NOCS_Predictor(cfg.NOCS_Predictor)
        self.estimator = PoseSizeEstimator()

        self.reconstructor = Reconstructor(cfg.Reconstructor)
        
        
    def forward(self, inputs):
        end_points = {}

        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        cls = inputs['category_label'].reshape(-1)

        c = torch.mean(pts, 1, keepdim=True)  
        pts = pts - c
        
        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.cat_num
        
        # rgb feat
        if self.cfg.rgb_backbone == 'resnet':
            rgb_local = self.rgb_extractor(rgb) 
        elif self.cfg.rgb_backbone == 'dino':
            dino_feature = self.rgb_extractor .forward_features(rgb)["x_prenorm"][:, 1:]  
            f_dim = dino_feature.shape[-1]
            num_patches =int(dino_feature.shape[1]**0.5)
            dino_feature = dino_feature.reshape(b, num_patches, num_patches, f_dim).permute(0,3,1,2)
            dino_feature = F.interpolate(dino_feature, size=(num_patches * 14, num_patches * 14), mode='bilinear', align_corners=False) 
            dino_feature = dino_feature.reshape(b, f_dim, -1) 
            rgb_local = self.feature_mlp(dino_feature)
        else:
            raise NotImplementedError
        
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous() # b, c, n

        if self.training:
            delta_r, delta_t, delta_s = generate_augmentation(b)
            pts = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r

        pts_local = self.pts_extractor(pts) # b, c, n
        batch_kpt_query, heat_map = self.IAKD(rgb_local, pts_local)
        kpt_3d = torch.bmm(heat_map, pts)
        kpt_feature = torch.bmm(heat_map, torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2))
        kpt_feature = self.GAFA(kpt_feature, kpt_3d.detach(), torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2), pts)
        recon_model, recon_delta = self.reconstructor(kpt_3d.transpose(1, 2), kpt_feature.transpose(1, 2))
        kpt_nocs = self.nocs_predictor(kpt_feature, index)
        r, t, s = self.estimator(kpt_3d, kpt_nocs.detach(), kpt_feature)

        if self.training:
            end_points['recon_delta'] = recon_delta
            end_points['pred_heat_map'] = heat_map
            end_points['pred_kpt_3d'] =  \
            (kpt_3d @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['recon_model'] =  \
            (recon_model.transpose(1, 2) @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['pred_kpt_nocs'] = kpt_nocs
            end_points['pred_translation'] = delta_t.squeeze(1) + delta_s * torch.bmm(delta_r, t.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation'] = delta_r @ r
            end_points['pred_size'] = s * delta_s

        else:
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_rotation'] = r
            end_points['pred_size'] = s
            end_points['pred_kpt_3d'] =  kpt_3d + c
            end_points['kpt_nocs'] = kpt_nocs

        return end_points

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def forward(self, endpoints):
        
        pts = endpoints['pts']
        b = pts.shape[0] 
        
        recon_delta = endpoints['recon_delta']
        pred_kpt_3d = endpoints['pred_kpt_3d']
        recon_model = endpoints['recon_model']
        
        translation_gt = endpoints['translation_label']
        rotation_gt = endpoints['rotation_label']
        size_gt = endpoints['size_label']
        
        # pose 
        loss_pose = PoseDis(endpoints['pred_rotation'], endpoints['pred_translation'], endpoints['pred_size'], rotation_gt, translation_gt, size_gt)
        # cd
        loss_cd = self.cd_dis_k2p(pts, pred_kpt_3d)
        # nocs
        kpt_nocs_gt = (pred_kpt_3d - translation_gt.unsqueeze(1)) / (torch.norm(size_gt, dim=1).view(b, 1, 1) + 1e-8) @ rotation_gt
        loss_nocs = SmoothL1Dis(endpoints['pred_kpt_nocs'], kpt_nocs_gt)
        # div
        loss_diversity = self.diversity_loss_3d(pred_kpt_3d)
        # reconstruction
        if self.cfg.obj_aware:
            # recon_with_mask
            loss_recon = self.ChamferDis_with_mask(pts, recon_model, endpoints['pc_mask'])
        else:
            loss_recon = ChamferDis(pts, recon_model)
        # regularization 
        loss_delta = recon_delta.norm(dim=2).mean()
        
        loss_all = self.cfg.pose*loss_pose + self.cfg.nocs*loss_nocs + self.cfg.cd*loss_cd + \
            self.cfg.diversity*loss_diversity + self.cfg.recon*loss_recon + self.cfg.delta*loss_delta
        return {
            'loss_all': loss_all,
            'loss_pose': self.cfg.pose*loss_pose,
            'loss_nocs': self.cfg.nocs*loss_nocs,
            'loss_cd': self.cfg.cd*loss_cd,
            'loss_diversity': self.cfg.diversity*loss_diversity,
            'loss_recon': self.cfg.recon*loss_recon,
            'loss_delta': self.cfg.delta*loss_delta,
        }
        
    def ChamferDis_with_mask(self, pts, recon_model, pc_mask):
        """
        calculate ChamferDis with valid pointcloud mask
        Args:
            pts: (b, n1, 3)
            recon_model: (b, n2, 3)
            pc_mask: (b, n1)

        Return:
            recon_loss
        """
        b = pts.shape[0]
        is_first = True
        
        for idx in range(b):
            pts_ = pts[idx] # (n1, 3)
            pts_ = pts_[pc_mask[idx] == True] 
            if pts_.shape[0] == 0:
                print('warning: no valid point')
                continue
            recon_model_ = recon_model[idx] 
            dis = ChamferDis_wo_Batch(pts_, recon_model_)
            
            if is_first:
                dis_all = dis
                is_first = False
            else:
                dis_all += dis
        return dis_all / b

    def cd_dis_k2p(self, pts, pred_kpt_3d):
        """_summary_

        Args:
            pts (_type_): (b, n, 3)
            pred_kpt_3d (_type_): (b, kpt_num, 3)
        """
        # (b, n, 1, 3)   -   (b, 1, kpt_num, 3)  = (b, n, kpt_num, 3) -> (b, n, kpt_num)
        dis = torch.norm(pts.unsqueeze(2) - pred_kpt_3d.unsqueeze(1), dim=3)
        # (b, kpt_num)
        dis = torch.min(dis, dim=1)[0]
        return torch.mean(dis)
    
    def diversity_loss_3d(self, data):
        """_summary_

        Args:
            data (_type_): (b, kpt_num, 3)
        """
        threshold = self.cfg.th
        b, kpt_num = data.shape[0], data.shape[1]
        
        dis_mat = data.unsqueeze(2) - data.unsqueeze(1)
        # (b, kpt_num, kpt_num)
        dis_mat = torch.norm(dis_mat, p=2, dim=3, keepdim=False)
        
        dis_mat = dis_mat + torch.eye(kpt_num, device=dis_mat.device).unsqueeze(0)
        dis_mat[dis_mat >= threshold] = threshold
    
        # dis=0 -> loss=1     dis=threshold -> loss=0   y= -x/threshold + 1
        dis_mat = -dis_mat / threshold + 1
        
        loss = torch.sum(dis_mat, dim=[1, 2])
        loss = loss / (kpt_num * (kpt_num-1))
        
        return loss.mean()