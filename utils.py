import os
import argparse
from datetime import datetime
import logging
from resnet_tiny import resnet56
from torchvision.models import resnet50
from torchvision import datasets, transforms as T

NORMALIZE_DICT = {
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
}

MODEL_DICT = {
    'resnet56': resnet56,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument("--mode", type=str, required=True, choices=["prune", "test"])
    parser.add_argument("--model", type=str, default='resnet56')
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar10', 'cifar100'])
    parser.add_argument('--dataroot', default='data', help='path to your datasets')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
    parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--restore", type=str, default="./cifar10_resnet56.pth")
    parser.add_argument('--output-dir', default='run', help='path where to save')
    parser.add_argument("--finetune", action="store_true", default=True, help='whether finetune or not')

    # For pruning
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--speed-up", type=float, default=2)
    parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--global-pruning", action="store_true", default=False)
    parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
    parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
    parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
    parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
    parser.add_argument("--sl-restore", type=str, default=None)
    parser.add_argument("--iterative-steps", default=200, type=int)
    parser.add_argument('--debug', type=str2bool, const=True, nargs='?',
                        default=False, help='When in the debug mode, it will not record logs')
    args = parser.parse_args()
    return args


def get_logger(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.mode == "prune":
        logger_name = "{}-{}-{}".format(args.dataset, args.method, args.model)
        log_file = "{}/{}.log".format(args.output_dir, logger_name)
    else:
        logger_name = None
        log_file = None
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    if logger_name is not None and not args.debug:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def get_dataset(name: str, data_root: str = 'data', return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser(data_root)

    if name == 'cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
        input_size = (1, 3, 32, 32)
    elif name == 'cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join(data_root, 'torchdata')
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
        input_size = (1, 3, 32, 32)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, input_size, train_transform, val_transform
    return num_classes, train_dst, val_dst, input_size


def get_model(name: str, num_classes, pretrained=False, target_dataset='cifar', **kwargs):
    if 'cifar' in target_dataset:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model


def flatten_dict(dic):
    flattned = dict()

    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten(k, v)
                else:
                    _flatten(prefix + '/%s' % k, v)
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[prefix + '/%s' % k] = v

    _flatten(None, dic)
    return flattned


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_remain_time(max_iter, cur_i, iter_time):
    remain_iter = max_iter - cur_i
    remain_time = remain_iter * iter_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return remain_time
