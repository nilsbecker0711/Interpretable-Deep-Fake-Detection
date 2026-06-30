"""Parse a training log file and plot per-epoch train/val/test metrics."""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


TRAIN_LOSS_RE = re.compile(r"training-loss, overall:\s*([\d.]+)")
TRAIN_METRIC_RE = re.compile(r"training-metric,\s*(\w+):\s*([\d.]+)")
VAL_LOSS_RE = re.compile(r"dataset: FaceForensics\+\+\s+step:.*?val-loss, overall:\s*([\d.]+)")
VAL_METRIC_RE = re.compile(r"val-metric,\s*(\w+):\s*([\d.]+)")
TEST_LOSS_RE = re.compile(r"dataset: FaceForensics\+\+\s+step:.*?test-loss, overall:\s*([\d.]+)")
TEST_METRIC_RE = re.compile(r"test-metric,\s*(\w+):\s*([\d.]+)")
EPOCH_START_RE = re.compile(r"===> Epoch\[(\d+)\] start!")
VAL_START_RE = re.compile(r"===> Val start!")
TEST_START_RE = re.compile(r"===> Test start!")
VAL_DONE_RE = re.compile(r"===> Val Done!")
TEST_DONE_RE = re.compile(r"===> Test Done!")


def parse_log(log_path: Path):
    train_losses = defaultdict(list)  # epoch -> [loss values]
    train_metrics = defaultdict(lambda: defaultdict(list))  # epoch -> metric -> [values]
    val_losses = {}   # epoch -> loss
    val_metrics = {}  # epoch -> {metric: value}
    test_losses = {}
    test_metrics = {}

    current_epoch = None
    in_val = False
    in_test = False

    with open(log_path) as f:
        for line in f:
            epoch_match = EPOCH_START_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                in_val = False
                in_test = False
                continue

            if VAL_START_RE.search(line):
                in_val = True
                in_test = False
                continue
            if VAL_DONE_RE.search(line):
                in_val = False
                continue
            if TEST_START_RE.search(line):
                in_test = True
                in_val = False
                continue
            if TEST_DONE_RE.search(line):
                in_test = False
                continue

            if in_val and current_epoch is not None:
                m = VAL_LOSS_RE.search(line)
                if m:
                    val_losses[current_epoch] = float(m.group(1))
                for name, value in VAL_METRIC_RE.findall(line):
                    val_metrics.setdefault(current_epoch, {})[name] = float(value)
                continue

            if in_test and current_epoch is not None:
                m = TEST_LOSS_RE.search(line)
                if m:
                    test_losses[current_epoch] = float(m.group(1))
                for name, value in TEST_METRIC_RE.findall(line):
                    test_metrics.setdefault(current_epoch, {})[name] = float(value)
                continue

            # training lines (outside val/test blocks)
            if current_epoch is not None:
                m = TRAIN_LOSS_RE.search(line)
                if m:
                    train_losses[current_epoch].append(float(m.group(1)))
                for name, value in TRAIN_METRIC_RE.findall(line):
                    train_metrics[current_epoch][name].append(float(value))

    # Average training metrics per epoch
    epochs_train = sorted(train_losses.keys())
    train_loss_avg = {e: np.mean(train_losses[e]) for e in epochs_train}
    train_metric_avg = {
        e: {k: np.mean(v) for k, v in train_metrics[e].items()}
        for e in epochs_train
    }

    return train_loss_avg, train_metric_avg, val_losses, val_metrics, test_losses, test_metrics


def plot_metrics(log_path: Path):
    train_loss, train_m, val_loss, val_m, test_loss, test_m = parse_log(log_path)

    # Collect all metric names
    all_metric_names = set()
    for d in [train_m, val_m, test_m]:
        for v in d.values():
            all_metric_names.update(v.keys())
    # Keep a sensible order
    metric_order = ["acc", "auc", "video_auc", "eer", "ap", "rc", "f1"]
    metrics = [m for m in metric_order if m in all_metric_names]
    metrics += sorted(all_metric_names - set(metrics))

    epochs_val = sorted(val_loss.keys())
    epochs_train = sorted(train_loss.keys())

    n_plots = 1 + len(metrics)  # 1 for loss + one per metric
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    def plot_series(ax, title, ylabel, train_vals, val_vals, test_vals):
        if train_vals:
            xs, ys = zip(*sorted(train_vals.items()))
            ax.plot(xs, ys, label="Train", marker="o", markersize=3, linewidth=1.5)
        if val_vals:
            xs, ys = zip(*sorted(val_vals.items()))
            ax.plot(xs, ys, label="Val", marker="s", markersize=3, linewidth=1.5)
        if test_vals:
            xs, ys = zip(*sorted(test_vals.items()))
            ax.plot(xs, ys, label="Test", marker="^", markersize=3, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Loss plot
    plot_series(
        axes[0], "Loss", "Loss",
        train_loss,
        val_loss,
        test_loss,
    )

    for i, metric in enumerate(metrics):
        train_vals = {e: v[metric] for e, v in train_m.items() if metric in v}
        val_vals = {e: v[metric] for e, v in val_m.items() if metric in v}
        test_vals = {e: v[metric] for e, v in test_m.items() if metric in v}
        plot_series(
            axes[i + 1],
            metric.upper().replace("_", " "),
            metric,
            train_vals,
            val_vals,
            test_vals,
        )

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    run_name = log_path.parent.name
    fig.suptitle(f"Training curves — {run_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = log_path.parent / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Default: look for training.log in the current directory
        log_path = Path("training.log")
        if not log_path.exists():
            print("Usage: python plot_training_log.py <path/to/training.log>")
            sys.exit(1)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    plot_metrics(log_path)
