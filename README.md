# Pruning ResNet-56 on CIFAR-10
This project conducts a series of pruning experiments for the pretrained model ResNet-56 on dataset CIFAR-10.
The all codes are modified based on the project [Torch-Pruning](https://github.com/VainF/Torch-Pruning) for better understanding.

## Previous methods and references
**L1**: [R1] Li. et al. Pruning Filters for Efficient ConvNets. ICLR 2017.

**BN**: [R2] Liu et al. Learning Efficient Convolutional Networks through Network Slimming. ICCV 2017.

**GR**: [R3] Wang et al. Neural Pruning via Growing Regularization. ICLR 2021.

**Group**: [R4] Fang et al. DepGraph: Towards Any Structural Pruning. CVPR 2023. 


## Compared methods
**Random**: Randomly prune specific ratio of weights for each layer.

**Group-L1**, **Group-BN** and **Group-GR**: combining Group method in [R4] with each other reference paper.

**Group-SL**: consistent sparse learning in [R4].

## Scripts to prune:
```shell
# 2.0x for Group-Random
python main.py --mode prune --batch-size 128 --dataset cifar10 --method random --speed-up 2.0
```

```shell
# 2.0x for Group-L1
python main.py --mode prune --batch-size 128 --dataset cifar10 --method l1 --speed-up 2.0 --global-pruning
```

```shell
# 2.0x for Group-BN
python main.py --mode prune --batch-size 128 --dataset cifar10  --method slim --speed-up 2.0 --global-pruning --reg 1e-5
```


```shell
# 2.0x for Group-GR
python main.py --mode prune --batch-size 128 --dataset cifar10  --method growing_reg --speed-up 2.0 --global-pruning --reg 1e-4 --delta_reg 1e-5
```

```shell
# 2.0x for Group-SL
python main.py --mode prune --batch-size 128 --dataset cifar10  --method group_sl --speed-up 2.0 --global-pruning --reg 5e-4
```
