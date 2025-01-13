import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
import time
import utils as u
from utils import parse_args, get_logger, get_dataset, get_model, flatten_dict


args = parse_args()


def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()


def train_model(
        model, train_loader, test_loader, epochs, lr, lr_decay_milestones, lr_decay_gamma=0.1, save_as=None,
        # For pruning
        weight_decay=5e-4, save_state_dict_only=True, pruner=None, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    model.to(device)
    best_acc = -1
    epoch_time = u.AverageMeter()
    for epoch in range(epochs):
        end = time.time()
        model.train()

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)  # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
            pruner.update_reg()  # increase the strength of regularization
            # print(pruner.group_reg[pruner._groups[0]])

        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        epoch_time.update(time.time() - end)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}, Epoch time {:.3f} {:.3f}, Remain {}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"],
                epoch_time.val, epoch_time.avg, u.calc_remain_time(epochs, epoch, epoch_time)
            )
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_{args.model}.pth")

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, "{}_{}.pth".format(args.dataset, args.model))
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))


def get_pruner(model, example_inputs):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning,
                               group_lasso=True)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max')  # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "growing_reg":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg,
                               global_pruning=args.global_pruning)
    else:
        raise NotImplementedError

    # args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)

    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model, example_inputs, importance=imp, iterative_steps=args.iterative_steps, pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict, max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers, unwrapped_parameters=unwrapped_parameters,
    )
    return pruner


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    args.output_dir = os.path.join(args.output_dir + f'_sp{args.speed_up}', args.dataset, args.mode)
    args.logger = get_logger(args)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = get_dataset(
        args.dataset, data_root=args.dataroot
    )
    args.num_classes = num_classes
    model = get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=8
    )

    for k, v in flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu", weights_only=True)
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)

    ######################################################
    # Pruning / Testing
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "prune":
        pruner = get_pruner(model, example_inputs=example_inputs)
        # 0. Sparsity Learning
        if args.sparsity_learning:
            reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
            reg_pth = os.path.join(os.path.join(args.output_dir, reg_pth))
            if not args.sl_restore:
                args.logger.info("Regularizing...")
                train_model(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.sl_total_epochs,
                    lr=args.sl_lr,
                    lr_decay_milestones=args.sl_lr_decay_milestones,
                    lr_decay_gamma=args.lr_decay_gamma,
                    pruner=pruner,
                    save_state_dict_only=True,
                    save_as=reg_pth,
                )
            args.logger.info("Loading the sparse model from {}...".format(reg_pth))
            model.load_state_dict(torch.load(reg_pth, map_location=args.device))

        # 1. Pruning
        model.eval()
        ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)
        args.logger.info("Pruning...")
        progressive_pruning(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs)
        del pruner  # remove reference
        args.logger.info(model)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)

        args.logger.info(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            )
        )
        args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6,
                pruned_ops / 1e6,
                pruned_ops / ori_ops * 100,
                ori_ops / pruned_ops,
            )
        )
        args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info(
            "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
        )

        # 2. Fine-tuning
        if args.finetune:
            args.logger.info("Finetuning...")
            train_model(
                model,
                epochs=args.total_epochs,
                lr=args.lr,
                lr_decay_milestones=args.lr_decay_milestones,
                train_loader=train_loader,
                test_loader=test_loader,
                device=args.device,
                save_state_dict_only=False,
            )
    elif args.mode == "test":
        model.eval()
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        acc, val_loss = eval(model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))


if __name__ == "__main__":
    main()
