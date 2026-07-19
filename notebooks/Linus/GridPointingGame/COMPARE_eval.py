#!/usr/bin/env python3
"""Evaluate several (model, XAI-method) combos on the SAME shared GPG grids.

The model list and all paths come from a yaml spec (see
compare_configs.example.yaml in this folder):

    python COMPARE_eval.py --compare-config compare_configs.example.yaml

CLI flags override the spec values, e.g.:

    python COMPARE_eval.py --compare-config my_compare.yaml \
        --grids-root results/GPG --output-dir results/GPG/compare --num-grids 4
"""
import os
import re
import sys
import glob
import pickle
import argparse
import collections
from collections import defaultdict

# set project path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

import torch
import yaml
from training.utils.xai.B_COS_eval import BCOSEvaluator
from training.utils.xai.LIME_eval import LIMEEvaluator
from training.utils.xai.GradCam_eval import GradCamEvaluator
from Utils_PointingGame import load_config, load_model
from training.utils.xai.xai_common import canonicalize_grid, adapt_for_model


def resolve_path(path):
    """Interpret relative paths as relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_PATH, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models/XAI methods on the same shared GPG grids")
    parser.add_argument("--compare-config", required=True,
                        help="Yaml spec with grids_root, output_dir, num_grids, "
                             "grid_subpath and the models list "
                             "(see compare_configs.example.yaml)")
    parser.add_argument("--grids-root", default=None,
                        help="Overrides grids_root: folder containing <gridpath>/<grid_subpath>/*.pt")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides output_dir for the result pickles")
    parser.add_argument("--num-grids", type=int, default=None,
                        help="Overrides num_grids: total number of grids to evaluate")
    return parser.parse_args()


def load_spec(args):
    # the spec file itself may be given relative to the shell's cwd;
    # everything inside it is resolved relative to the repo root.
    spec_path = args.compare_config if os.path.exists(args.compare_config) \
        else resolve_path(args.compare_config)
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    if args.grids_root is not None:
        spec["grids_root"] = args.grids_root
    if args.output_dir is not None:
        spec["output_dir"] = args.output_dir
    if args.num_grids is not None:
        spec["num_grids"] = args.num_grids

    for key in ("grids_root", "output_dir", "models"):
        if not spec.get(key):
            raise ValueError(f"Missing '{key}' (set it in the spec or via CLI)")
    spec.setdefault("num_grids", 2)
    spec.setdefault("grid_subpath", "3x3")
    for cfg in spec["models"]:
        for key in ("name", "gridpath", "model_yaml", "run_yaml", "xai"):
            if not cfg.get(key):
                raise ValueError(f"Model entry {cfg.get('name', cfg)} is missing '{key}'")
    return spec


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def collect_shared_grids(spec, device):
    """For each unique gridpath, pick the same grids once, shared by all models."""
    gridpath_to_configs = defaultdict(list)
    for cfg in spec["models"]:
        gridpath_to_configs[cfg["gridpath"]].append(cfg)

    unique_gridpaths = list(gridpath_to_configs.keys())
    N = spec["num_grids"]
    U = len(unique_gridpaths)
    K = max(1, N // U)

    grid_cache = {}
    shared_grid_paths = []

    for gridpath in unique_gridpaths:
        folder = os.path.join(resolve_path(spec["grids_root"]), gridpath, spec["grid_subpath"])

        pts = sorted(glob.glob(os.path.join(folder, "*.pt")), key=natural_key)
        print(f"→ Found {len(pts)} files in {folder}.")
        if not pts:
            raise FileNotFoundError(f"No grid .pt files in {folder} — run GPG_eval grid "
                                    f"creation first or fix grids_root/gridpath")

        selected = pts[:K]
        shared_grid_paths += selected

        for path in selected:
            if path not in grid_cache:
                grid_cache[path] = torch.load(path, map_location=device)

    shared_grids = [grid_cache[p] for p in shared_grid_paths]

    print("\n Shared grids selected for evaluation:")
    for i, path in enumerate(shared_grid_paths):
        print(f"  [{i+1:02}] {path}")
    print(f"\n→ Total: {len(shared_grid_paths)} grid(s) used across all model configs.")

    assert len(shared_grids) == len(shared_grid_paths), "Mismatch in loaded grids"
    return shared_grids, shared_grid_paths


def evaluate_model(cfg, shared_grids, shared_grid_paths, spec, device):
    print(f"\n=== Evaluating {cfg['name']} ===")

    # a) load config + instantiate model
    config = load_config(resolve_path(cfg["model_yaml"]),
                         resolve_path(cfg["run_yaml"]), additional_args={})
    model = load_model(config)
    model.to(device).eval()

    # b) load & inject pretrained weights ('weights' in the spec wins over the
    #    'pretrained' key from the yamls)
    wpath = cfg.get("weights") or config.get("pretrained")
    if not wpath:
        raise ValueError(f"{cfg['name']}: no checkpoint — set 'weights' in the spec "
                         f"or 'pretrained' in the yamls")
    sd = torch.load(resolve_path(wpath), map_location="cpu")
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v
    res = model.load_state_dict(new_sd, strict=False)
    print("   missing_keys:   ", res.missing_keys)
    print("   unexpected_keys:", res.unexpected_keys)

    # c) pick your evaluator
    if cfg["xai"] == "bcos":
        evaluator = BCOSEvaluator(model, device)
    elif cfg["xai"] in ["gradcam", "xgrad", "grad++", "layergrad"]:
        evaluator = GradCamEvaluator(model, device, method=cfg["xai"])
    elif cfg["xai"] == "lime":
        evaluator = LIMEEvaluator(model, device,
                                  mean=config.get("mean", [0.5, 0.5, 0.5]),
                                  std=config.get("std", [0.5, 0.5, 0.5]))
    else:
        raise RuntimeError(f"Unknown XAI '{cfg['xai']}'")

    print(f"Preparing grids for XAI method: {cfg['xai']}")

    # d) per-MODEL input adaptation (scale AND channels).
    # Grids are canonical raw [0,1] RGB; each model gets its own preprocessing:
    # standard models -> mean/std normalization, b-cos models -> [x, 1-x].
    # Grids stored in a normalized value range are detected and denormalized
    # with a loud warning (stale asset).
    mean = config.get("mean", [0.5, 0.5, 0.5])
    std = config.get("std", [0.5, 0.5, 0.5])
    grids = []
    for i, (g, gpath) in enumerate(zip(shared_grids, shared_grid_paths)):
        g_canon = canonicalize_grid(g, mean=mean, std=std, warn_name=gpath)
        g_out = adapt_for_model(g_canon, model, mean=mean, std=std)
        print(f"  Grid {i}: {tuple(g.shape)} → canonical {tuple(g_canon.shape)} "
              f"→ model input {tuple(g_out.shape)}")
        grids.append(g_out)

    # e) run the pointing-game evaluation
    raw = evaluator.evaluate(
        tensor_list    = grids,
        path_list      = shared_grid_paths,
        grid_split     = config["grid_split"],
        threshold_steps= config.get("threshold_steps", 0),
    )

    # f) group & pickle raw results by threshold
    threshold_groups = collections.defaultdict(list)
    for entry in raw:
        thr = entry.get("threshold", None)
        threshold_groups[thr].append(entry)

    out_dir = resolve_path(spec["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    all_raw_path = os.path.join(
        out_dir,
        f"results_by_threshold_{cfg['name']}_{cfg['xai']}.pkl"
    )
    with open(all_raw_path, "wb") as f:
        pickle.dump(dict(threshold_groups), f)
    print(f" → saved grouped raw results: {all_raw_path}")


def main():
    args = parse_args()
    spec = load_spec(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_grids, shared_grid_paths = collect_shared_grids(spec, device)

    for cfg in spec["models"]:
        evaluate_model(cfg, shared_grids, shared_grid_paths, spec, device)


if __name__ == "__main__":
    main()
