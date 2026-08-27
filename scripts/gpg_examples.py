#!/usr/bin/env python
"""Qualitative GPG figures: which 3x3 cell does each XAI method point at?

    python scripts/gpg_examples.py --num-grids 5

One row per grid: the grid itself with the true fake cell outlined, then one
panel per (model, method) with the attribution overlaid, annotated with the
weighted localization score. Chance is 1/9 = 0.111.

Uses the same evaluators the GPG runs use, so the maps and numbers match.
"""
import argparse
import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import logging                                          # noqa: E402
logging.disable(logging.INFO)

from detectors import DETECTOR                                          # noqa: E402
from training.utils.xai.B_COS_eval import BCOSEvaluator, evaluate_heatmap  # noqa: E402
from training.utils.xai.GradCam_eval import GradCamEvaluator            # noqa: E402
from training.utils.xai.IG_eval import IGEvaluator                      # noqa: E402
from training.utils.xai.xai_common import canonicalize_grid, adapt_for_model  # noqa: E402

PANELS = [
    ("xception",            "logs/training/xception_2*/val/avg/ckpt_best.pth",
     ["gradcam", "grad++", "ig"]),
    ("xception_bcos_b1_75", "logs/training/xception_bcos_detector_b1_75_*/val/avg/ckpt_best.pth",
     ["bcos", "gradcam", "grad++", "ig"]),
]

GRID_DIR = "results/GPG_assets/shared_random/FaceForensics++_test_256/3x3"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CMAP = LinearSegmentedColormap.from_list(
    "attr", [(0, (0, 0, 0, 0)), (0.35, (0.8, 0.1, 0.0, 0.45)),
             (0.7, (1.0, 0.5, 0.0, 0.8)), (1, (1.0, 1.0, 0.2, 0.95))])


def load(cfg_name, pat):
    cfg = yaml.safe_load(open(f"training/config/detector/{cfg_name}.yaml"))
    cfg.update(yaml.safe_load(open("training/config/test_config.yaml")))
    model = DETECTOR[cfg["model_name"]](cfg).to(DEV)
    ck = sorted(glob.glob(pat), reverse=True)
    if not ck:
        raise SystemExit(f"no checkpoint for {cfg_name}")
    sd = torch.load(ck[0], map_location=DEV)
    model.load_state_dict(
        OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()), strict=False)
    model.eval()
    return model, cfg


def evaluator_for(method, model, cfg):
    if method == "bcos":
        return BCOSEvaluator(model, DEV)
    if method == "ig":
        return IGEvaluator(model, DEV)
    return GradCamEvaluator(model, DEV, method=method)


def fake_pos(path):
    """The grid filename encodes the true fake cell: ..._fake_<idx>_conf..."""
    base = os.path.basename(path)
    return int(base.split("_fake_")[1].split("_")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-grids", type=int, default=5)
    ap.add_argument("--grid-dir", default=GRID_DIR)
    ap.add_argument("-o", "--out", default="results/eval/mpg_examples/GPG_examples.png")
    args = ap.parse_args()

    paths = [os.path.join(args.grid_dir, f)
             for f in sorted(os.listdir(args.grid_dir)) if f.endswith(".pt")][:args.num_grids]
    print(f"{len(paths)} grids from {args.grid_dir}")

    rows, cols = {}, []
    rgb_of = {}
    for cfg_name, pat, methods in PANELS:
        print(f"  {cfg_name}: {', '.join(methods)}")
        model, cfg = load(cfg_name, pat)
        mean, std = cfg.get("mean", [0.5] * 3), cfg.get("std", [0.5] * 3)
        for meth in methods:
            ev = evaluator_for(meth, model, cfg)
            for p in paths:
                t = adapt_for_model(
                    canonicalize_grid(torch.load(p, map_location=DEV),
                                      mean=mean, std=std, warn_name=p),
                    model, mean=mean, std=std)
                t = ev.prepare_input(t) if hasattr(ev, "prepare_input") else t
                if p not in rgb_of:
                    im = t[:, :3].squeeze().detach().cpu().permute(1, 2, 0).numpy()
                    rgb_of[p] = (im - im.min()) / (im.max() - im.min() + 1e-8)
                hm = ev.generate_heatmap(t if meth != "gradcam" and meth != "grad++"
                                         and meth != "xgrad" and meth != "layergrad"
                                         else t.squeeze(0))[0]
                hm = np.asarray(hm, dtype=np.float32)
                _, _, w, _, _ = evaluate_heatmap(hm, grid_split=3, true_fake_pos=fake_pos(p))
                rows[(cfg_name, meth, p)] = (hm, float(w))
            del ev
            cols.append((cfg_name, meth))
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    ncol = 1 + len(cols)
    fig, axes = plt.subplots(len(paths), ncol, figsize=(2.05 * ncol, 2.2 * len(paths)))
    axes = np.atleast_2d(axes)
    short = {"xception": "Xception", "xception_bcos_b1_75": "B-cos b=1.75"}
    for r, p in enumerate(paths):
        pos = fake_pos(p)
        fr, fc = pos % 3, pos // 3      # filename index is column-major (see GPG_eval)
        for c in range(ncol):
            ax = axes[r][c]
            ax.imshow(rgb_of[p])
            if c > 0:
                cfg_name, meth = cols[c - 1]
                hm, w = rows[(cfg_name, meth, p)]
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                ax.imshow(hm, cmap=CMAP, vmin=0, vmax=1)
                ax.text(0.02, 0.02, f"{w:.2f}", transform=ax.transAxes, fontsize=8,
                        color="white", va="bottom",
                        bbox=dict(fc="black", alpha=0.65, pad=1.4, lw=0))
                if r == 0:
                    ax.set_title(f"{short.get(cfg_name, cfg_name)}\n{meth}", fontsize=8)
            elif r == 0:
                ax.set_title("grid (cyan = fake cell)", fontsize=9)
            H = rgb_of[p].shape[0]
            cell = H / 3
            for k in (1, 2):
                ax.axhline(k * cell, color="white", lw=0.4, alpha=0.5)
                ax.axvline(k * cell, color="white", lw=0.4, alpha=0.5)
            ax.add_patch(plt.Rectangle((fc * cell, fr * cell), cell, cell,
                                       fill=False, ec="cyan", lw=1.6))
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Grid Pointing Game -- shared FF++ grids   "
                 "(cyan = cell holding the fake, number = weighted score, chance 0.111)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(paths)} grids x {ncol} panels)")


if __name__ == "__main__":
    main()
