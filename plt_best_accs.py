import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def parse_log(file_path):
    final_accs, final_ratios = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Best Acc='):
                best_acc = float(line.split('=')[-1].strip())
                final_accs.append(best_acc)
            elif line.startswith('FLOPs: '):
                items = line.split(',')[-1].strip()
                final_ratio = float(items.split('X')[0].strip())
                final_ratios.append(final_ratio)
    return final_accs[-1], final_ratios[-1]


if __name__ == '__main__':
    sps = [2.0, 4.0, 8.0, 16.0]
    best_data = {
        "random": [],
        "l1": [],
        "slim": [],
        "growing_reg": [],
        "group_sl": [],
    }
    legends = ['Random', 'Group-L1', 'Group-BN', 'Group-GR', 'Group-SL']
    for sp in sps:
        fp = Path(f'run_sp{sp:.1f}/cifar10/prune/')
        for mn in best_data.keys():
            cur_best_acc, cur_ratio = parse_log(str(fp / f"cifar10-{mn}-resnet56.log"))
            best_data[mn].append((cur_best_acc, cur_ratio))
    plt.figure(figsize=(8, 6))

    for di, data in enumerate(best_data.items()):
        label = legends[di]
        ratios, accs = [], []
        for spi in range(len(data[1])):
            accs.append(data[1][spi][0])
            ratios.append(data[1][spi][1])
        linestyle = "-" if "Group" in label else "--"
        plt.plot(ratios, accs, label=label, linestyle=linestyle)

    plt.xticks(ticks=[2, 4, 8, 16], labels=[2, 4, 8, 16])
    plt.xlabel("Speed Up", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy vs. Speed Up", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
