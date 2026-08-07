"""Parse a training log file and plot per-epoch train/val/test metrics."""

import re
import sys
import pickle
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

# In-training XAI monitor run folders: "<model_name>_<method>_epoch<NNNN>".
# model_name itself contains underscores, so match the known method suffix.
XAI_DIR_RE = re.compile(
    r"^(?P<model>.+)_(?P<method>bcos|gradcam|xgrad|grad\+\+|layergrad|lime)_epoch(?P<epoch>\d+)$"
)
# Parent folder of overall_by_threshold.pkl -> which game produced it.
XAI_GAME_BY_SUBDIR = {"3x3": "GPG", "MaskPointingGame": "MPG"}
XAI_METRICS = ("weighted", "unweighted")

# Operating points overlaid for the post-training TEST results. The in-training
# monitor always writes threshold 0 only (threshold_steps=0), so this applies to
# --final-xai-dir alone.
FINAL_XAI_THRESHOLDS = (0.0, 0.2, 0.4, 0.6, 0.8)


def _mean_scores(pkl_path, threshold=0):
    """Return {metric: mean localization score at `threshold`} for one game/run,
    or {} if the file is unreadable or lacks that operating point.
    overall_by_threshold.pkl maps threshold -> {'<metric>_localization_score':
    [per-grid scores], ...}.

    Threshold keys are written as i/threshold_steps, so they are matched with a
    tolerance rather than by exact equality — never rely on float keys hashing
    to the value you typed.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return {}
    entry = data.get(threshold)
    if entry is None:  # tolerant lookup: nearest stored threshold within 1e-6
        for key in data:
            if isinstance(key, (int, float)) and abs(float(key) - threshold) < 1e-6:
                entry = data[key]
                break
    if entry is None:
        return {}
    out = {}
    for metric in XAI_METRICS:
        scores = entry.get(f"{metric}_localization_score")
        if scores is not None and len(scores):
            out[metric] = float(np.mean(scores))
    return out


def parse_xai_games(base_dir: Path):
    """Per-epoch val localization curves written by the in-training XAI monitor.

    Returns {game: {metric: {method: {epoch: mean_score}}}} for
    game in {'GPG','MPG'}, metric in {'weighted','unweighted'}. Empty (but
    well-formed) if the monitor never ran — the caller degrades gracefully.
    """
    out = {g: {m: defaultdict(dict) for m in XAI_METRICS}
           for g in XAI_GAME_BY_SUBDIR.values()}
    xai_root = base_dir / "val" / "xai_games"
    if not xai_root.is_dir():
        return out
    for run_dir in sorted(xai_root.iterdir()):
        m = XAI_DIR_RE.match(run_dir.name)
        if not m:
            continue
        method, epoch = m.group("method"), int(m.group("epoch"))
        for subdir, game in XAI_GAME_BY_SUBDIR.items():
            # The in-training monitor runs with threshold_steps=0, so threshold 0
            # is the only operating point it ever writes.
            means = _mean_scores(run_dir / subdir / "overall_by_threshold.pkl", 0)
            for metric, value in means.items():
                out[game][metric][method][epoch] = value
    return out


def parse_final_xai(final_dir: Path, thresholds=FINAL_XAI_THRESHOLDS):
    """Post-training XAI test results to overlay as reference lines.

    Recursively finds every overall_by_threshold.pkl under final_dir; the game
    comes from the immediate parent folder (3x3 / MaskPointingGame) and the
    label from its grandparent (the eval run's own folder name). Returns
    {game: {metric: {'<label> @ t=0.20': score}}}.

    One line per (run, threshold): the final-eval configs sweep thresholds
    (threshold_steps: 10), and the ordering between XAI methods is NOT stable
    across the sweep — a low threshold counts a CAM's positive floor as mass,
    which flatters CAM methods relative to the sparser B-cos maps. Reporting a
    single operating point hides that.
    """
    out = {g: {m: {} for m in XAI_METRICS} for g in XAI_GAME_BY_SUBDIR.values()}
    if final_dir is None or not final_dir.is_dir():
        return out
    for pkl in sorted(final_dir.rglob("overall_by_threshold.pkl")):
        game = XAI_GAME_BY_SUBDIR.get(pkl.parent.name)
        if game is None:
            continue
        run = pkl.parent.parent.name
        for t in thresholds:
            for metric, value in _mean_scores(pkl, t).items():
                out[game][metric][f"{run} @ t={t:.2f}"] = value
    return out


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


def plot_xai_series(ax, title, per_method, chance=None, final=None):
    """One XAI subplot: a per-epoch val curve per method, optional chance-level
    line and post-training final-test reference lines."""
    plotted = False
    for method, series in sorted(per_method.items()):
        if not series:
            continue
        xs, ys = zip(*sorted(series.items()))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=f"{method} (val)")
        plotted = True
    if chance is not None:
        ax.axhline(chance, color="gray", linestyle=":", linewidth=1,
                   label=f"chance ({chance:.3f})")
    if final:
        for label, score in sorted(final.items()):
            ax.axhline(score, linestyle="--", linewidth=1.2, alpha=0.85,
                       label=f"final test: {label} ({score:.3f})")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("localization score")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return plotted


def plot_metrics(log_path: Path, final_xai_dir: Path = None,
                 final_xai_thresholds=FINAL_XAI_THRESHOLDS):
    train_loss, train_m, val_loss, val_m, test_loss, test_m = parse_log(log_path)
    xai = parse_xai_games(log_path.parent)
    final_xai = parse_final_xai(final_xai_dir, final_xai_thresholds)

    # Collect all metric names
    all_metric_names = set()
    for d in [train_m, val_m, test_m]:
        for v in d.values():
            all_metric_names.update(v.keys())
    # Keep a sensible order
    metric_order = ["acc", "auc", "video_auc", "eer", "ap", "rc", "f1"]
    metrics = [m for m in metric_order if m in all_metric_names]
    metrics += sorted(all_metric_names - set(metrics))

    # XAI localization subplots (only those with data). GPG has a 1/9 chance line.
    xai_specs = [
        ("GPG", "weighted", "GPG weighted (val)", 1.0 / 9.0),
        ("GPG", "unweighted", "GPG unweighted (val)", 1.0 / 9.0),
        ("MPG", "weighted", "MPG weighted (val)", None),
        ("MPG", "unweighted", "MPG unweighted (val)", None),
    ]
    active_xai = [
        s for s in xai_specs
        if any(xai[s[0]][s[1]].values()) or any(final_xai[s[0]][s[1]].values())
    ]

    epochs_val = sorted(val_loss.keys())
    epochs_train = sorted(train_loss.keys())

    n_plots = 1 + len(metrics) + len(active_xai)  # loss + metrics + XAI
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

    # XAI localization subplots after loss + accuracy metrics
    xai_offset = 1 + len(metrics)
    for k, (game, metric, title, chance) in enumerate(active_xai):
        plot_xai_series(
            axes[xai_offset + k],
            title,
            xai[game][metric],
            chance=chance,
            final=final_xai[game][metric],
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot per-epoch train/val/test metrics + in-training XAI "
                    "localization curves (GPG/MPG) from a training run.")
    parser.add_argument("log_path", nargs="?", default="training.log",
                        help="Path to training.log (XAI pickles are read from "
                             "its sibling val/xai_games/ folder).")
    parser.add_argument("--final-xai-dir", default=None,
                        help="Optional folder with post-training GPG/MPG TEST "
                             "results (overall_by_threshold.pkl files); each is "
                             "overlaid as a reference line on the matching subplot.")
    parser.add_argument("--final-xai-thresholds", type=float, nargs="+",
                        default=list(FINAL_XAI_THRESHOLDS), metavar="T",
                        help="Attribution thresholds to overlay from --final-xai-dir, "
                             "one reference line each (default: %(default)s). Only "
                             "operating points present in the pickles are drawn. The "
                             "in-training curves are always threshold 0.")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("Usage: python plot_training_log.py <path/to/training.log> "
              "[--final-xai-dir <dir>]")
        sys.exit(1)

    final_dir = Path(args.final_xai_dir) if args.final_xai_dir else None
    plot_metrics(log_path, final_dir, args.final_xai_thresholds)
