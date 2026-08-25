#!/usr/bin/env python3
"""Sanity checks for the XAI localisation pipeline.

Run this after any change to the evaluators, the grid assets or the dataset
classes. Every check either PASSES or FAILS with the numbers that justify the
verdict; the exit code is the number of failures, so it can gate a commit.

    # model-free checks only (no GPU, safe to run during training)
    python training/utils/xai/sanity_check.py

    # add the model-dependent checks (repeat --model per model)
    python training/utils/xai/sanity_check.py \
        --model resnet34.yaml:logs/training/<run>/val/avg/ckpt_best.pth \
        --model resnet34_bcos_v2_b2.yaml:logs/training/<run>/val/avg/ckpt_best.pth

While a training run owns the GPU, --memory-fraction caps this process so it
cannot eat the run's headroom (ViT at 768 grids needs ~6 GiB: 48x48 = 2304
tokens, and attention is quadratic in that).

WHAT EACH CHECK DEFENDS AGAINST -- every one of these corresponds to a bug that
actually occurred in this pipeline:
  metric semantics   a row/column transpose between grid building and scoring
  threshold-0        the unweighted score being 1/9 by construction, not by merit
  top-k budget       tie-keeping silently giving coarse maps a larger pixel budget
  asset ground truth the fake image not being in the cell the filename claims
  split hygiene      val and test assets sharing source images
  mask loading       masks loading blank on one dataset path but not the other
  position control   preprocessing / target-layer errors that survive a static grid
  completeness       a b-cos map that is not the model's own decomposition
  reference match    our GradCAM drifting from the B-cos-v2 algorithm
  chance level       a broken pipeline scoring above chance for the wrong reason
  token order        a ViT reshape_transform that scrambles tokens into pixels
"""
import os
import sys
import glob
import json
import argparse
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (PROJECT_PATH, os.path.join(PROJECT_PATH, "training")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import numpy as np

from training.utils.xai.GradCam_eval import evaluate_heatmap
from training.utils.xai.xai_common import topn_mask, smooth_map, normalize_max

CHANCE = 1.0 / 9.0
DEFAULT_GRID_DIR = os.path.join(
    PROJECT_PATH, "results/GPG_assets/shared_random/FaceForensics++_test_256/3x3")

FAILURES = []


def check(name, passed, detail=""):
    print(("  PASS  " if passed else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not passed:
        FAILURES.append(name)
    return passed


# ---------------------------------------------------------------------------
# model-free checks
# ---------------------------------------------------------------------------

def check_metric_semantics():
    """The metric's cell indexing must agree with how grids are built."""
    print("\n=== metric semantics ===")
    H = 768
    bad = []
    for cell in range(9):
        m = np.zeros((H, H), np.float32)
        r, c = divmod(cell, 3)
        m[r*256:(r+1)*256, c*256:(c+1)*256] = 1.0
        pw, _, wa, pu, ua = evaluate_heatmap(m, 3, true_fake_pos=cell)
        if not (pw == cell and pu == cell and abs(wa - 1) < 1e-6 and abs(ua - 1) < 1e-6):
            bad.append(cell)
    check("all mass in cell k scores 1.0 at cell k, for every k (row-major)",
          not bad, f"cells failing: {bad}" if bad else "cells 0..8")

    _, _, wa, _, ua = evaluate_heatmap(np.ones((H, H), np.float32), 3, true_fake_pos=4)
    check("uniform map scores exactly chance, weighted and unweighted",
          abs(wa - CHANCE) < 1e-6 and abs(ua - CHANCE) < 1e-6, f"w={wa:.4f} u={ua:.4f}")

    rng = np.random.default_rng(0)
    m = rng.random((H, H)).astype(np.float32) * 0.9 + 0.1
    m[512:768, 512:768] = 1.0
    _, _, wa, _, ua = evaluate_heatmap(m, 3, true_fake_pos=8)
    check("KNOWN ARTIFACT: any strictly positive map is exactly 1/9 unweighted",
          abs(ua - CHANCE) < 1e-9, f"u={ua:.6f}, while weighted {wa:.3f} discriminates")

    _, _, wa_t, _, _ = evaluate_heatmap(np.where(m < 0.95, 0.0, m), 3, true_fake_pos=8)
    check("thresholding a peaked map raises its weighted score", wa_t > wa,
          f"{wa:.3f} -> {wa_t:.3f}")


def check_topk_budget():
    """top-k keeps ties on purpose; verify that does not skew the budget by method."""
    print("\n=== top-k pixel budget is comparable across map structures ===")
    import torch
    import torch.nn.functional as F
    H, frac = 768, 0.025
    budget = int(round(frac * H * H))
    rng = np.random.default_rng(0)
    kept = {}

    for feat in (24, 48):  # CAM: coarse map, NEAREST upsampled -> large tied blocks
        t = torch.from_numpy(rng.random((1, 1, feat, feat)).astype(np.float32))
        up = F.interpolate(t, size=(H, H), mode="nearest")[0, 0].numpy()
        kept[f"cam{feat}"] = int((topn_mask(normalize_max(up), frac) > 0).sum())
    kept["bcos"] = int((topn_mask(normalize_max(np.clip(
        smooth_map(rng.standard_normal((H, H)).astype(np.float32), 15), 0, None)), frac) > 0).sum())
    seg = rng.integers(0, 50, size=(H // 16, H // 16))
    lime = np.kron(rng.random(50).astype(np.float32)[seg], np.ones((16, 16), np.float32))
    kept["lime(smoothed as in LIME_eval)"] = int(
        (topn_mask(normalize_max(smooth_map(lime, 15)), frac) > 0).sum())

    worst = max(kept.values()) / budget
    check("every map structure gets ~the same pixel budget", worst < 1.15,
          " ".join(f"{k}={v/budget:.2f}x" for k, v in kept.items()))


def check_grid_assets(grid_dir, n_grids):
    """The fake image must really sit in the cell the filename and manifest claim."""
    print(f"\n=== grid asset ground truth ({os.path.basename(os.path.dirname(grid_dir))}) ===")
    import torch
    from PIL import Image

    man_path = os.path.join(grid_dir, "manifest.json")
    if not os.path.exists(man_path):
        check("grid manifest exists", False, man_path)
        return
    entries = {g["file"]: g for g in json.load(open(man_path))["grids"]}
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))[:n_grids]
    if not files:
        check("grid folder holds tensors", False, grid_dir)
        return

    def corr(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d else 0.0

    name_bad = cell_bad = missing = range_bad = 0
    for f in files:
        base = os.path.basename(f)
        ent = entries[base]
        if int(base.split("_fake_")[1].split("_conf_")[0]) != ent["fake_position"]:
            name_bad += 1
            continue
        t = torch.load(f, map_location="cpu")
        t = t[0] if t.dim() == 4 else t
        res = t.shape[-1] // 3
        if not (-0.01 <= float(t.min()) and float(t.max()) <= 1.01):
            range_bad += 1
        if not os.path.exists(ent["fake_image"]):
            missing += 1
            continue
        ref = np.asarray(Image.open(ent["fake_image"]).convert("RGB").resize((res, res)),
                         np.float32).transpose(2, 0, 1) / 255.
        scores = []
        for i in range(9):
            r, c = divmod(i, 3)
            scores.append(corr(t[:3, r*res:(r+1)*res, c*res:(c+1)*res].numpy(), ref))
        if int(np.argmax(scores)) != ent["fake_position"]:
            cell_bad += 1

    n = len(files)
    check(f"filename position == manifest position ({n} grids)", name_bad == 0, f"{name_bad} bad")
    check(f"the fake SOURCE IMAGE sits in the declared cell ({n} grids)", cell_bad == 0,
          f"{cell_bad} bad")
    check("every fake source image still exists on disk", missing == 0, f"{missing} missing")
    check("grids are stored in canonical [0,1]", range_bad == 0, f"{range_bad} out of range")


def check_split_hygiene(test_dir):
    """val and test assets must not share source images."""
    print("\n=== val/test asset hygiene ===")
    val_dir = test_dir.replace("_test_", "_val_")
    if not os.path.exists(os.path.join(val_dir, "manifest.json")):
        print("  SKIP  no matching val asset folder")
        return
    t = json.load(open(os.path.join(test_dir, "manifest.json")))["grids"]
    v = json.load(open(os.path.join(val_dir, "manifest.json")))["grids"]
    fo = {g["fake_image"] for g in t} & {g["fake_image"] for g in v}
    ro = ({p for g in t for p in g["real_images"]} & {p for g in v for p in g["real_images"]})
    check("val and test share no fake source image", not fo, f"overlap {len(fo)}")
    check("val and test share no real source image", not ro, f"overlap {len(ro)}")


def check_mask_loading(model_yaml="resnet34.yaml"):
    """MPG masks must load non-blank on BOTH dataset paths (they are separate code)."""
    print("\n=== MPG mask loading, both dataset paths ===")
    import torch
    sys.path.insert(0, os.path.join(PROJECT_PATH, "notebooks/Linus/GridPointingGame"))
    from Utils_PointingGame import load_config

    def prepare():
        argv = sys.argv
        sys.argv = [sys.argv[0]]          # train.py parses argv at import time
        try:
            from train import prepare_testing_data
        finally:
            sys.argv = argv
        return prepare_testing_data

    for label, dtype in (("standard (abstract_dataset)", ""), ("b-cos (b_cos_pp)", "bcos")):
        cfg = load_config(
            os.path.join(PROJECT_PATH, "training/config/detector", model_yaml),
            os.path.join(PROJECT_PATH, "training/config/test_config.yaml"),
            additional_args={"test_batchSize": 8, "with_mask": True,
                             "dataset_type": dtype, "test_dataset": ["FaceForensics++"]})
        try:
            loaders = prepare()(cfg)
        except Exception as exc:
            check(f"{label}: loader builds", False, f"{type(exc).__name__}: {exc}")
            continue
        key = list(loaders.keys())[0]
        fakes = blank = absent = 0
        for batch in loaders[key]:
            masks = batch.get("mask")
            for j in range(batch["image"].shape[0]):
                if int(batch["label"][j]) == 0:
                    continue
                fakes += 1
                if masks is None:
                    absent += 1
                elif float(masks[j].max()) == 0:
                    blank += 1
            if fakes >= 40:
                break
        check(f"{label}: fake masks load non-blank", fakes > 0 and blank == 0 and absent == 0,
              f"{fakes} fakes, {blank} blank, {absent} absent, "
              f"{type(loaders[key].dataset).__name__}")


# ---------------------------------------------------------------------------
# model-dependent checks
# ---------------------------------------------------------------------------

def _load_model(yaml_name, weights, device):
    import torch
    import yaml as _yaml
    from collections import OrderedDict
    from training.detectors import DETECTOR

    path = yaml_name if os.path.isabs(yaml_name) else os.path.join(
        PROJECT_PATH, "training/config/detector", yaml_name)
    with open(path) as f:
        cfg = _yaml.safe_load(f)
    cfg.setdefault("label_dict", {"FF-real": 0, "FF-fake": 1})
    cfg["pretrained"] = None
    model = DETECTOR[cfg["model_name"]](cfg)
    info = "random weights"
    if weights:
        wpath = weights if os.path.isabs(weights) else os.path.join(PROJECT_PATH, weights)
        sd = torch.load(wpath, map_location="cpu")
        res = model.load_state_dict(
            OrderedDict((k.replace("module.", ""), v) for k, v in sd.items()), strict=False)
        info = f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    return model.to(device).eval(), cfg, info


def _methods_for(model):
    return ["gradcam", "bcos"] if hasattr(model.backbone, "explain") else ["gradcam"]


def _evaluator(model, method, device):
    from training.utils.xai.B_COS_eval import BCOSEvaluator
    from training.utils.xai.GradCam_eval import GradCamEvaluator
    return (BCOSEvaluator(model, device) if method == "bcos"
            else GradCamEvaluator(model, device, method=method))


def _heat(ev, method, t):
    return np.asarray(ev.generate_heatmap(t if method == "bcos" else t.squeeze(0))[0])


def _grid(path, model, cfg, device):
    import torch
    from training.utils.xai.xai_common import canonicalize_grid, adapt_for_model
    mean, std = cfg.get("mean", [0.5]*3), cfg.get("std", [0.5]*3)
    g = canonicalize_grid(torch.load(path, map_location="cpu"), mean=mean, std=std)
    g = g if g.dim() == 4 else g.unsqueeze(0)
    return adapt_for_model(g, model, mean=mean, std=std).to(device), g


def check_position_control(spec, grid_dir, device):
    """Rotate the fake cell through all 9 positions; attribution must follow it.

    The strongest end-to-end check: grid construction, per-model preprocessing,
    target-layer resolution and metric indexing all have to agree for this to pass.
    """
    import torch
    from training.utils.xai.xai_common import adapt_for_model
    print("\n=== position control (fake rotated through all 9 cells) ===")
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))
    man = {g["file"]: g for g in json.load(open(os.path.join(grid_dir, "manifest.json")))["grids"]}
    base_f = files[0]
    fpos = man[os.path.basename(base_f)]["fake_position"]

    for yaml_name, weights in spec:
        model, cfg, info = _load_model(yaml_name, weights, device)
        mean, std = cfg.get("mean", [0.5]*3), cfg.get("std", [0.5]*3)
        _, canon = _grid(base_f, model, cfg, device)
        res = canon.shape[-1] // 3
        cells = [canon[..., r*res:(r+1)*res, c*res:(c+1)*res]
                 for r in range(3) for c in range(3)]
        fake_cell = cells[fpos]
        reals = [c for i, c in enumerate(cells) if i != fpos]

        for method in _methods_for(model):
            ev = _evaluator(model, method, device)
            hits, scores = 0, []
            for pos in range(9):
                grid = torch.zeros(1, canon.shape[1], res*3, res*3, dtype=canon.dtype)
                it = iter(reals)
                for i in range(9):
                    r, c = divmod(i, 3)
                    grid[..., r*res:(r+1)*res, c*res:(c+1)*res] = fake_cell if i == pos else next(it)
                t = adapt_for_model(grid, model, mean=mean, std=std).to(device)
                pred, _, wa, _, _ = evaluate_heatmap(_heat(ev, method, t), 3, true_fake_pos=pos)
                hits += int(pred == pos)
                scores.append(wa)
                del t
                torch.cuda.empty_cache()
            check(f"{os.path.basename(yaml_name)}/{method}: attribution follows the fake cell",
                  hits >= 6, f"{hits}/9 positions, mean weighted {np.mean(scores):.3f} "
                             f"(chance {CHANCE:.3f}) [{info}]")
        del model
        torch.cuda.empty_cache()


def check_completeness(spec, grid_dir, device):
    """A b-cos contribution map must sum to the class logit -- it IS the decomposition."""
    import torch
    print("\n=== b-cos completeness ===")
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))
    ran = False
    for yaml_name, weights in spec:
        model, cfg, _ = _load_model(yaml_name, weights, device)
        if hasattr(model.backbone, "explain"):
            ran = True
            t, _ = _grid(files[0], model, cfg, device)
            t = t.requires_grad_(True)
            expl = model.backbone.explain(t, idx=1)
            cm = expl["contribution_map"][0]
            csum = float(cm.sum()) if torch.is_tensor(cm) else float(np.sum(cm))
            with torch.no_grad():
                logit = float(model({"image": t.detach()})["cls"][0, 1])
            rel = abs(csum - logit) / max(abs(logit), 1e-8)
            check(f"{os.path.basename(yaml_name)}: sum(contribution_map) == logit[fake]",
                  rel < 0.05, f"sum={csum:.4f} logit={logit:.4f} rel gap={rel:.2e}")
            del t
        del model
        torch.cuda.empty_cache()
    if not ran:
        print("  SKIP  no b-cos model in the given spec")


def check_reference_gradcam(spec, grid_dir, device):
    """Our GradCAM must reproduce the B-cos-v2 / captum algorithm on the same layer."""
    import torch
    from captum.attr import LayerGradCam
    from training.utils.xai.GradCam_eval import GradCamEvaluator, WrappedModel
    print("\n=== GradCAM vs the reference algorithm ===")
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))
    man = {g["file"]: g for g in json.load(open(os.path.join(grid_dir, "manifest.json")))["grids"]}
    fp = man[os.path.basename(files[0])]["fake_position"]

    for yaml_name, weights in spec:
        model, cfg, _ = _load_model(yaml_name, weights, device)
        if hasattr(model.backbone, "gradcam_reshape_transform"):
            # captum's LayerGradCam cannot take a reshape hook, so it averages a
            # (B, N, D) token tensor as if D were spatial -- the same limitation
            # that makes method='layergrad' unsupported on ViT.
            print(f"  SKIP  {os.path.basename(yaml_name)}: token model, captum "
                  f"LayerGradCam cannot serve as a reference here")
            del model
            continue
        t, _ = _grid(files[0], model, cfg, device)
        ev = GradCamEvaluator(model, device, method="gradcam")
        ours = _heat(ev, "gradcam", t)
        attr = LayerGradCam(WrappedModel(model), ev.target_layer).attribute(
            t.clone().requires_grad_(True), target=1, relu_attributions=True)
        ref = LayerGradCam.interpolate(attr, tuple(t.shape[-2:]), interpolate_mode="nearest")
        ref = np.clip(ref[0].sum(0).detach().cpu().numpy(), 0, None)
        ref = ref / (ref.max() + 1e-8)
        a, b = ours.ravel() - ours.mean(), ref.ravel() - ref.mean()
        corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        _, _, wo, _, _ = evaluate_heatmap(ours, 3, true_fake_pos=fp)
        _, _, wr, _, _ = evaluate_heatmap(ref, 3, true_fake_pos=fp)
        check(f"{os.path.basename(yaml_name)}: GradCAM matches the reference", corr > 0.9,
              f"pixel corr {corr:.3f} | GPG ours {wo:.4f} vs ref {wr:.4f}")
        del model, t
        torch.cuda.empty_cache()


def check_determinism(spec, grid_dir, device):
    import torch
    print("\n=== determinism ===")
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))
    for yaml_name, weights in spec:
        model, cfg, _ = _load_model(yaml_name, weights, device)
        t, _ = _grid(files[0], model, cfg, device)
        for method in _methods_for(model):
            ev = _evaluator(model, method, device)
            d = float(np.abs(_heat(ev, method, t) - _heat(ev, method, t)).max())
            check(f"{os.path.basename(yaml_name)}/{method}: repeated call is identical",
                  d < 1e-6, f"max|diff|={d:.2e}")
        del model, t
        torch.cuda.empty_cache()


def check_chance_and_signal(spec, grid_dir, device, n_grids):
    """Untrained must sit at chance; trained must beat it."""
    import torch
    print("\n=== chance (random weights) vs signal (trained) ===")
    files = sorted(glob.glob(os.path.join(grid_dir, "*.pt")))[:n_grids]
    man = {g["file"]: g for g in json.load(open(os.path.join(grid_dir, "manifest.json")))["grids"]}
    for yaml_name, weights in spec:
        for tag, w in (("untrained", None), ("trained", weights)):
            model, cfg, _ = _load_model(yaml_name, w, device)
            for method in _methods_for(model):
                ev = _evaluator(model, method, device)
                sc = []
                for f in files:
                    t, _ = _grid(f, model, cfg, device)
                    _, _, wa, _, _ = evaluate_heatmap(
                        _heat(ev, method, t), 3,
                        true_fake_pos=man[os.path.basename(f)]["fake_position"])
                    sc.append(wa)
                    del t
                    torch.cuda.empty_cache()
                mu = float(np.mean(sc))
                label = f"{os.path.basename(yaml_name)}/{method} {tag}"
                if tag == "untrained":
                    check(f"{label} sits at chance", abs(mu - CHANCE) < 0.06,
                          f"mean {mu:.4f} vs chance {CHANCE:.4f}, n={len(files)}")
                else:
                    check(f"{label} beats chance", mu > CHANCE + 0.02,
                          f"mean {mu:.4f} vs chance {CHANCE:.4f}, n={len(files)}")
            del model
            torch.cuda.empty_cache()


def check_token_order(spec, device):
    """ViT only: the reshape must map tokens back to the right image region.

    Probed with a DIFFERENCE against a neutral input -- raw activation norm is
    dominated by a position-independent residual-stream component and localises
    nothing, so an absolute-magnitude probe would report noise.
    """
    import torch
    from training.utils.xai.xai_common import adapt_for_model
    print("\n=== ViT token -> pixel mapping ===")
    ran = False
    for yaml_name, weights in spec:
        model, cfg, _ = _load_model(yaml_name, weights, device)
        bb = model.backbone
        if not hasattr(bb, "gradcam_reshape_transform"):
            del model
            continue
        ran = True
        mean, std = cfg.get("mean", [0.5]*3), cfg.get("std", [0.5]*3)
        hits, side = 0, None
        for cell in range(9):
            g = torch.full((1, 3, 768, 768), 0.5)
            r, c = divmod(cell, 3)
            # BLACK, not noise: uniform noise has mean 0.5 -- the same as the
            # neutral baseline -- so the difference signal would be weak and the
            # result would vary run to run. A black patch is a real mean shift.
            g[..., r*256:(r+1)*256, c*256:(c+1)*256] = 0.0
            t = adapt_for_model(g, model, mean=mean, std=std).to(device)
            tb = adapt_for_model(torch.full((1, 3, 768, 768), 0.5), model,
                                 mean=mean, std=std).to(device)
            store = {}
            h = bb.transformer.register_forward_hook(lambda m, i, o: store.__setitem__("z", o))
            with torch.no_grad():
                model({"image": t});  zp = store["z"].clone()
                model({"image": tb}); zb = store["z"].clone()
            h.remove()
            energy = bb.gradcam_reshape_transform(zp - zb)[0].abs().mean(0)
            side = energy.shape[-1]
            s = side // 3
            per_cell = [float(energy[i*s:(i+1)*s, j*s:(j+1)*s].mean())
                        for i in range(3) for j in range(3)]
            hits += int(int(np.argmax(per_cell)) == cell)
            del t, tb, zp, zb
            torch.cuda.empty_cache()
        check(f"{os.path.basename(yaml_name)}: token order and grid side are correct",
              hits >= 8, f"{hits}/9 cells, token grid {side}x{side}")
        del model
        torch.cuda.empty_cache()
    if not ran:
        print("  SKIP  no token model in the given spec")


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[], metavar="YAML:WEIGHTS",
                   help="Detector yaml and checkpoint, e.g. resnet34.yaml:logs/.../ckpt_best.pth. "
                        "Repeatable. Without any, only the model-free checks run.")
    p.add_argument("--grid-dir", default=DEFAULT_GRID_DIR,
                   help="Shared GPG grid folder to audit (default: the test_256 asset)")
    p.add_argument("--grids", type=int, default=12,
                   help="Grids per model-dependent check (default 12)")
    p.add_argument("--asset-grids", type=int, default=30,
                   help="Grids for the asset ground-truth audit (default 30)")
    p.add_argument("--memory-fraction", type=float, default=0.25,
                   help="Cap this process' share of GPU memory so a concurrent "
                        "training run keeps its headroom (default 0.25)")
    p.add_argument("--skip-masks", action="store_true",
                   help="Skip the dataset mask check (it builds full data loaders)")
    return p.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    check_metric_semantics()
    check_topk_budget()
    check_grid_assets(args.grid_dir, args.asset_grids)
    check_split_hygiene(args.grid_dir)
    if not args.skip_masks:
        check_mask_loading()

    spec = []
    for entry in args.model:
        if ":" not in entry:
            raise SystemExit(f"--model expects YAML:WEIGHTS, got {entry!r}")
        y, w = entry.split(":", 1)
        spec.append((y, w))

    if spec:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"\nGPU memory capped at {args.memory_fraction*total:.1f} GiB "
                  f"(a concurrent training run keeps the rest)")
        check_position_control(spec, args.grid_dir, device)
        check_completeness(spec, args.grid_dir, device)
        check_reference_gradcam(spec, args.grid_dir, device)
        check_determinism(spec, args.grid_dir, device)
        check_chance_and_signal(spec, args.grid_dir, device, args.grids)
        check_token_order(spec, device)
    else:
        print("\n(no --model given: model-dependent checks skipped)")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES")
        for f in FAILURES:
            print("  -", f)
    else:
        print("ALL SANITY CHECKS PASSED")
    return len(FAILURES)


if __name__ == "__main__":
    sys.exit(main())
