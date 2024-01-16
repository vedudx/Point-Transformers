# Pytorch Implementation of Various Point Transformers

Recently, various methods applied transformers to point clouds: [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688), [Point Transformer (Nico Engel et al.)](https://arxiv.org/abs/2011.00931), [Point Transformer (Hengshuang Zhao et al.)](https://arxiv.org/abs/2012.09164). This repo is a pytorch implementation for these methods and aims to compare them under a fair setting. Currently, all three methods are implemented, while tuning their hyperparameters.


## Classification
### Data Preparation
Prepare data in the same format as ['model net 40'](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).

### Requirement
install CUDA supported pytorch, and then 
```
pip install -r requirements.txt
```
### Help
Most of the scripts have instructions on usage, so please refer to those when facing any issues.

### Data Preparation
```
# Have a data directory
- bash scripts/augment.sh "original_folder" "new_folder"
- original_folder is the directory of your data folder, and new folder is what you want your new directory prefix to be
```
### Run
Change which method to use in `config/cls.yaml` and run
```
python train_cls.py
```

### Miscellaneous

Forked and Modified from [point transformer] (https://github.com/qq456cvb/Point-Transformers).
Some code and training settings are borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
Code for [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688) is adapted from the author's Jittor implementation https://github.com/MenghaoGuo/PCT.

