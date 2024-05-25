from nocs_dataset import TrainingDataset
from housecat6d_dataset import HouseCat6DTrainingDataset
import torch

def create_dataloaders(cfg):
    data_dir = cfg.dataset_dir
    data_loader = {}

    if cfg.dataset_name == "camera_real":
        syn_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.syn_bs, threshold=cfg.outlier_th)
            
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset,
            batch_size=cfg.syn_bs,
            num_workers=cfg.syn_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
            
        real_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'real_withLabel',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.real_bs, threshold=cfg.outlier_th)
            
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.real_bs,
            num_workers=cfg.real_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)

        data_loader['syn'] = syn_dataloader
        data_loader['real'] = real_dataloader
    
    elif cfg.dataset_name == "camera":
        syn_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.syn_bs, threshold=cfg.outlier_th)
            
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset,
            batch_size=cfg.syn_bs,
            num_workers=cfg.syn_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
            
        data_loader['syn'] = syn_dataloader
    
    elif cfg.dataset_name == "housecat6d":
        real_dataset = HouseCat6DTrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, cfg.seq_length, cfg.img_length)
        
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.batchsize,
            num_workers=int(cfg.num_workers),
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
        
        data_loader['real'] = real_dataloader
        
    else:
        raise NotImplementedError
    
    return data_loader