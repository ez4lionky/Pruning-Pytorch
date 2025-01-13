#python main.py --mode prune --batch-size 256 --dataset cifar10 --method random --speed-up 2.0
#python main.py --mode prune --batch-size 256 --dataset cifar10 --method l1 --speed-up 2.0 --global-pruning
#python main.py --mode prune --batch-size 256 --dataset cifar10  --method slim --speed-up 2.0 --global-pruning --reg 1e-5
#python main.py --mode prune --batch-size 256 --dataset cifar10  --method growing_reg --speed-up 2.0 --global-pruning --reg 1e-4 --delta_reg 1e-5
#python main.py --mode prune --batch-size 256 --dataset cifar10  --method group_sl --speed-up 2.0 --global-pruning --reg 5e-4

python main.py --mode prune --batch-size 256 --dataset cifar10 --method random --speed-up 4.0
python main.py --mode prune --batch-size 256 --dataset cifar10 --method l1 --speed-up 4.0 --global-pruning
python main.py --mode prune --batch-size 256 --dataset cifar10  --method slim --speed-up 4.0 --global-pruning --reg 1e-5
python main.py --mode prune --batch-size 256 --dataset cifar10  --method growing_reg --speed-up 4.0 --global-pruning --reg 1e-4 --delta_reg 1e-5
python main.py --mode prune --batch-size 256 --dataset cifar10  --method group_sl --speed-up 4.0 --global-pruning --reg 5e-4

python main.py --mode prune --batch-size 256 --dataset cifar10 --method random --speed-up 8.0
python main.py --mode prune --batch-size 256 --dataset cifar10 --method l1 --speed-up 8.0 --global-pruning
python main.py --mode prune --batch-size 256 --dataset cifar10  --method slim --speed-up 8.0 --global-pruning --reg 1e-5
python main.py --mode prune --batch-size 256 --dataset cifar10  --method growing_reg --speed-up 8.0 --global-pruning --reg 1e-4 --delta_reg 1e-5
python main.py --mode prune --batch-size 256 --dataset cifar10  --method group_sl --speed-up 8.0 --global-pruning --reg 5e-4

python main.py --mode prune --batch-size 256 --dataset cifar10 --method random --speed-up 16.0
python main.py --mode prune --batch-size 256 --dataset cifar10 --method l1 --speed-up 16.0 --global-pruning
#python main.py --mode prune --batch-size 256 --dataset cifar10  --method slim --speed-up 16.0 --global-pruning --reg 1e-5
python main.py --mode prune --batch-size 256 --dataset cifar10  --method growing_reg --speed-up 16.0 --global-pruning --reg 1e-4 --delta_reg 1e-5
#python main.py --mode prune --batch-size 256 --dataset cifar10  --method group_sl --speed-up 16.0 --global-pruning --reg 5e-4
