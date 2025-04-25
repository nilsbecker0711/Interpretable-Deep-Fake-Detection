#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import torch
import collections
from B_COS_eval import BCOSEvaluator
from GradCam_eval import GradCamEvaluator
from LIME_eval import LIMEEvaluator
from Utils_PointingGame import load_config, load_model

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
N = 4  # total number of grids to evaluate
model_configs = [
    {
        "name":        "resnet34_bcos_v2_minimal",
        "model_yaml":  "training/config/detector/resnet34_bcos_v2_minimal.yaml",
        "run_yaml":    "results/test_run3_config.yaml",
        "weights_key": "pretrained",
        "xai":         "bcos",
    },
    {
        "name":        "resnet34",
        "model_yaml":  "training/config/detector/resnet34.yaml",
        "run_yaml":    "results/test_run1_config.yaml",
        "weights_key": "pretrained",
        "xai":         "gradcam",
    },
    # … add more models here …
]
grid_subpath = "3x3/grids"      # relative inside each results folder
OUTPUT_BASE_DIR = "results/comparemode"
# ────────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) figure out how many grids per model
M = len(model_configs)
K = N // M

# 2) gather up to K grids from each model’s own results folder
shared_grid_paths = []
for cfg in model_configs:
    # each model’s grids live under results/{name}_{runname}/{grid_subpath}
    runname = os.path.splitext(os.path.basename(cfg["run_yaml"]))[0]
    folder = f"results/{cfg['name']}/{grid_subpath}"
    pts = sorted(glob.glob(os.path.join(folder, "*.pt")))
    shared_grid_paths += pts[:K]

# now you have exactly M * K == N grid files
assert len(shared_grid_paths) == K * M, f"expected {K*M}, got {len(shared_grid_paths)}"

# 3) load the actual tensors once
shared_grids = [ torch.load(p, map_location=device) for p in shared_grid_paths ]

# 4) loop over each model, load weights, evaluate on *the same* shared_grids
for cfg in model_configs:
    print(f"\n=== Evaluating {cfg['name']} ===")

    # a) load config + instantiate model
    config = load_config(cfg["model_yaml"], cfg["run_yaml"], additional_args={})
    model  = load_model(config)
    model.to(device).eval()

    # b) load & inject pretrained weights
    wpath = config[cfg["weights_key"]]
    sd    = torch.load(wpath, map_location="cpu")
    new_sd = {}
    for k,v in sd.items():
        new_sd[k.replace("module.","")] = v
    res = model.load_state_dict(new_sd, strict=False)
    print("   missing_keys:   ", res.missing_keys)
    print("   unexpected_keys:", res.unexpected_keys)

    # c) pick your evaluator
    if   cfg["xai"] == "bcos":    evaluator = BCOSEvaluator(model, device)
    elif cfg["xai"] == "gradcam": evaluator = GradCamEvaluator(model, device)
    elif cfg["xai"] == "lime":    evaluator = LIMEEvaluator(model, device)
    else: raise RuntimeError(f"Unknown XAI '{cfg['xai']}'")

    # d) drop to 3 channels if needed
    if cfg["xai"] in ("lime", "gradcam"):
        grids = []
        for g in shared_grids:
            # if [1,6,H,W] or [6,H,W]
            if g.ndim == 4: grids.append(g[:, :3, ...])
            elif g.ndim == 3: grids.append(g[:3, ...])
            else: raise ValueError(f"unexpected grid shape {g.shape}")
    else:
        grids = shared_grids

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

    out_dir = os.path.join(OUTPUT_BASE_DIR, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    all_raw_path = os.path.join(out_dir, "results_by_threshold.pkl")
    with open(all_raw_path, "wb") as f:
        pickle.dump(dict(threshold_groups), f)
    print(f" → saved grouped raw results: {all_raw_path}")

#python notebooks/Linus/GridPointingGame/COMPARE_eval.py