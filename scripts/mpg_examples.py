#!/usr/bin/env python
"""Qualitative MPG figures: ground-truth mask vs what each XAI method highlights.

    python scripts/mpg_examples.py --dataset FF-DF --num-images 6

One row per image: the face, the manipulation mask, then one panel per
(model, method) with the attribution map overlaid and the mask outline drawn on
top, annotated with that panel's weighted MPG score.

Reuses MaskPointingGameCreator.generate_heatmap_for_method and mask_game, so the
maps and the numbers are produced by exactly the code the MPG runs use -- this
script only arranges and draws them.
"""
import argparse
import json
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
sys.path.insert(0, os.path.join(REPO, "notebooks", "Linus", "GridPointingGame"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import logging                                        # noqa: E402
logging.disable(logging.INFO)

from MPG_eval import MaskPointingGameCreator, canon_image_path   # noqa: E402
from detectors import DETECTOR                                   # noqa: E402

# (config, checkpoint glob, methods) -- standard nets have no `bcos` explanation.
PANELS = [
    ("xception",            "logs/training/xception_2*/val/avg/ckpt_best.pth",
     ["gradcam", "grad++"]),
    ("xception_bcos_b1_75", "logs/training/xception_bcos_detector_b1_75_*/val/avg/ckpt_best.pth",
     ["bcos", "gradcam", "grad++"]),
]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transparent -> warm, so the face stays readable underneath the attribution
CMAP = LinearSegmentedColormap.from_list(
    "attr", [(0, (0, 0, 0, 0)), (0.35, (0.8, 0.1, 0.0, 0.45)),
             (0.7, (1.0, 0.5, 0.0, 0.8)), (1, (1.0, 1.0, 0.2, 0.95))])


def build(cfg_name, ckpt_glob, dataset, num_images, image_list, thresh_steps=0):
    import glob
    cfg = yaml.safe_load(open(f"training/config/detector/{cfg_name}.yaml"))
    cfg.update(yaml.safe_load(open("training/config/test_config.yaml")))
    cfg.update({
        "test_dataset": [dataset], "with_mask": True, "mask_resolution": 256,
        "max_images": num_images, "overwrite": True, "quantitativ": False,
        "threshold_steps": thresh_steps, "xai_method": "gradcam",
        "base_output_dir": "results/eval/mpg_examples", "test_batchSize": 1,
        "frame_num": {"train": 32, "test": 32, "val": 32},
    })
    model = DETECTOR[cfg["model_name"]](cfg).to(DEV)
    ck = sorted(glob.glob(ckpt_glob), reverse=True)
    if not ck:
        raise SystemExit(f"no checkpoint for {cfg_name} ({ckpt_glob})")
    sd = torch.load(ck[0], map_location=DEV)
    model.load_state_dict(
        OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()), strict=False)
    model.eval()

    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from train import prepare_testing_data
    finally:
        sys.argv = _argv
    loaders = prepare_testing_data(cfg)
    loader = list(loaders.values())[0]

    creator = MaskPointingGameCreator(
        base_output_dir=cfg["base_output_dir"], xai_method="gradcam", model=model,
        model_name=cfg["model_name"], config_name="examples",
        test_data_loaders=loaders, dataset=loader.dataset, device=DEV, config=cfg,
        overwrite=True, quantitativ=False, threshold_steps=0,
        max_images=num_images, mask_resolution=256, image_list=image_list)
    return creator, loader, cfg


def harvest(cfg_name, ckpt_glob, methods, dataset, wanted, num_images):
    """{image_path: {"rgb":…, "mask":…, method: (map, score)}} for one model."""
    creator, loader, cfg = build(cfg_name, ckpt_glob, dataset, num_images, wanted)
    out = {}
    for batch in loader:
        img, lab, mask = batch["image"], batch["label"], batch["mask"]
        paths = batch["image_path"]
        for j in range(img.shape[0]):
            p = paths[j] if not isinstance(paths[j], list) else paths[j][0]
            key = canon_image_path(str(p))
            if key not in wanted or key in out:
                continue
            if int(lab[j]) == 0:
                continue
            m = mask[j].squeeze()
            if m.shape != torch.Size([256, 256]) or torch.max(m) == 0:
                continue
            one = img[j:j + 1].to(DEV)
            rgb = one[:, :3].squeeze().detach().cpu().permute(1, 2, 0).numpy()
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rec = {"rgb": rgb, "mask": m.detach().cpu().numpy()}
            for meth in methods:
                hm = creator.generate_heatmap_for_method(meth, one.clone())
                hm = np.asarray(hm, dtype=np.float32)
                _, weighted = creator.mask_game(m, hm)
                rec[meth] = (hm, float(weighted))
                if meth == "bcos":
                    # The native B-cos RGBA rendering (gradient_to_image): colour
                    # encodes WHICH input channels contributed, alpha the strength.
                    # This is the per-pixel explanation itself; the scored map above
                    # is that map summed over channels and box-smoothed, which is
                    # what makes it look coarse. Kept for the figure only -- never
                    # scored (its alpha is percentile-clipped).
                    from training.utils.xai.B_COS_eval import BCOSEvaluator
                    rgba = BCOSEvaluator(creator.model, DEV).generate_heatmap(
                        one.clone())[1]
                    rec["bcos_rgba"] = np.asarray(rgba, dtype=np.float32)
            out[key] = rec
            # Collect the WHOLE candidate set, not the first num_images: the two
            # models iterate their own loaders in different orders (the b-cos
            # dataset shuffles at build time), so stopping early left them with
            # disjoint subsets and almost nothing to compare.
            if len(out) >= len(wanted):
                break
        if len(out) >= len(wanted):
            break
    del creator
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FF-DF")
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--image-list", default=None,
                    help="MPG image-list json; defaults to the newest run's list "
                         "so the examples come from the scored sample")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    lst = args.image_list
    if lst is None:
        import glob
        cands = sorted(glob.glob(
            f"results/eval/mpg/*/image_lists/{args.dataset}_images.json"), reverse=True)
        lst = cands[0] if cands else None
    if not lst or not os.path.exists(lst):
        raise SystemExit(f"no image list for {args.dataset}; pass --image-list")
    wanted_all = [canon_image_path(p) for p in json.load(open(lst))["images"]]
    # Slack for images either model skips (blank/mis-shaped mask, misclassified).
    wanted = set(wanted_all[: args.num_images * 3])
    print(f"image list: {lst}  ({len(wanted_all)} images)")

    harvested, cols = {}, []
    for cfg_name, ckpt, methods in PANELS:
        print(f"  {cfg_name}: {', '.join(methods)}")
        harvested[cfg_name] = harvest(cfg_name, ckpt, methods, args.dataset,
                                      wanted, args.num_images)
        for m in methods:
            cols.append((cfg_name, m))
            # Show the native RGBA explanation right next to the map that is
            # actually scored, so the smoothing cost is visible side by side.
            if m == "bcos":
                cols.append((cfg_name, "bcos_rgba"))

    common = [k for k in wanted_all if all(k in harvested[p[0]] for p in PANELS)]
    common = common[: args.num_images]
    if not common:
        raise SystemExit("no image was scored by both models")

    ncol = 2 + len(cols)
    fig, axes = plt.subplots(len(common), ncol,
                             figsize=(2.05 * ncol, 2.25 * len(common)))
    axes = np.atleast_2d(axes)
    short = {"xception": "Xception", "xception_bcos_b1_75": "B-cos b=1.75"}
    for r, key in enumerate(common):
        rec0 = harvested[PANELS[0][0]][key]
        mask = rec0["mask"]
        axes[r][0].imshow(rec0["rgb"])
        axes[r][1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r][0].set_title("fake image", fontsize=9)
            axes[r][1].set_title("manipulation mask", fontsize=9)
        axes[r][0].set_ylabel(os.path.basename(os.path.dirname(key)) + "/" +
                              os.path.basename(key), fontsize=6)
        for c, (cfg_name, meth) in enumerate(cols):
            ax = axes[r][2 + c]
            rec = harvested[cfg_name][key]
            ax.imshow(rec["rgb"])
            if meth == "bcos_rgba":
                # RGBA straight from the model: alpha composites it over the face.
                ax.imshow(np.clip(rec["bcos_rgba"], 0, 1))
                title = f"{short.get(cfg_name, cfg_name)}\nexplanation (RGBA)"
            else:
                hm, score = rec[meth]
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                ax.imshow(hm, cmap=CMAP, vmin=0, vmax=1)
                ax.text(0.03, 0.03, f"{score:.2f}", transform=ax.transAxes,
                        fontsize=8, color="white", va="bottom",
                        bbox=dict(fc="black", alpha=0.6, pad=1.4, lw=0))
                title = f"{short.get(cfg_name, cfg_name)}\n{meth}"
            ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.9)
            if r == 0:
                ax.set_title(title, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Mask Pointing Game -- {args.dataset}   "
                 f"(cyan = mask outline, number = weighted MPG score)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or f"results/eval/mpg_examples/{args.dataset}_examples.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(common)} images x {ncol} panels)")


if __name__ == "__main__":
    main()
