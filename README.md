# AG-Pose: Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation
This is the official implementation of extended version of CVPR24 paper "Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation"

[[Arxiv](https://arxiv.org/abs/2403.19527)]

The extended version primarily includes the following additions,

1. Add a reconstruction network to reconstruct input point clouds using detected keypoints.
2. Include experiments on [HouseCat6D (CVPR 2024 Highlight)](https://sites.google.com/view/housecat6d) dataset.
3. Include experiments using [DINOv2](https://github.com/facebookresearch/dinov2) as image backbone.

We will soon release a preprint about the extended paper where you can find more details.

## Citation
```
@article{lin2024instance,
  title={Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation},
  author={Lin, Xiao and Yang, Wenfei and Gao, Yuan and Zhang, Tianzhu},
  journal={arXiv preprint arXiv:2403.19527},
  year={2024}
}
```

## Environment Settings
The code has been tested with

- python 3.9
- torch 1.12
- cuda 11.3

Some dependencies:
```
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```
## Data Processing
### NOCS dataset
- Download and preprocess the dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN)
- Download and unzip the segmentation results [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip)

Put them under ```PROJ_DIR/data```and the final file structure is as follows:
```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   ├── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```
### HouseCat6D
Download and unzip the dataset from [HouseCat6D](https://sites.google.com/view/housecat6d) and the final file structure is as follows:
```
HOUSECAT6D_DIR
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```
## Train
### Training on NOCS
```
python train.py --config config/REAL/camera_real.yaml
```
### Training on HouseCat6D
```
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluate 
- Evaluate on NOCS:
```
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```
- Evaluate on HouseCat6D:
```
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```
## Results
You can download our training logs, detailed metrics for each category and checkpoints [here](http://home.ustc.edu.cn/~llinxiao/log.zip).
### REAL275 test set:

|   | IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|---|
| resnet_backbone | 84.3 | 83.8 | 77.6 | 56.2 | 62.3 | 73.4 | 81.2 |
| dino_backbone | 84.3 | 84.1 | 80.1 | 57.0 | 64.6 | 75.1 | 84.7 |

### CAMERA25 test set:

|   | IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|---|
| resnet_backbone | 94.7 | 94.1 | 91.7 | 77.1 | 82.0 | 85.5 | 91.6 |
| dino_backbone | 94.7 | 94.2 | 92.5 | 79.5 | 83.7 | 87.1 | 92.6 |

### HouseCat6D test set:

|   | IoU25 | IoU50 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|---|
| resnet_backbone | 82.4 | 66.0 | 40.5 | 11.5 | 12.6 | 37.4 | 42.5 |
| dino_backbone | 88.1 | 76.9 | 53.0 | 21.3 | 22.1 | 51.3 | 54.3 |
## Visualization
For visualization, please run
```
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```

## Acknowledgements
Our implementation leverages the code from these works:
- [NOCS](https://github.com/hughw19/NOCS_CVPR2019)
- [SPD](https://github.com/mentian/object-deformnet)
- [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)
- [DPDN](https://github.com/JiehongLin/Self-DPDN)
- [VI-Net](https://github.com/JiehongLin/VI-Net)
- [HouseCat6D Toolbox](https://github.com/Junggy/HouseCat6D)

We appreciate their generous sharing.
## License
Our code is released under MIT License (see LICENSE file for details).
## Contact
<llinxiao@mail.ustc.edu.cn>
