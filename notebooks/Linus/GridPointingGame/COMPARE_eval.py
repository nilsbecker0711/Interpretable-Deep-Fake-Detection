#!/usr/bin/env python3
import os
import re
import sys
import glob
import pickle
import torch
import collections
from collections import defaultdict
from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  
#from GradCam_evalnew import GradCamEvaluator
from GradCam_evallayer import GradCamEvaluator
#from GradCam_evalbcos import GradCamEvaluator
from Utils_PointingGame import load_config, load_model

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
N = 2  # total number of grids to evaluate
model_configs = [
    #{
    #    "name":        "resnet34_bcos_v2",
    #    "gridpath":    "resnet34_bcos_v2_1_25",
    #    "model_yaml":  "training/config/detector/resnet34_bcos_v2_1_25_best_hpo.yaml",
    #    "run_yaml":    "results/test_bcos_res_2_5_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "bcos",
    #},
    #{
    #    "name":        "resnet34_bcos_v2",
    #    "gridpath":    "resnet34_bcos_v2_2",
    #    "model_yaml":  "training/config/detector/resnet34_bcos_v2_2_best_hpo.yaml",
    #    "run_yaml":    "results/test_bcos_res_2_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "bcos",
    #},
    #{
    #    "name":        "resnet34_bcos_v2",
    #    "gridpath":    "resnet34_bcos_v2_2_5",
    #    "model_yaml":  "training/config/detector/resnet34_bcos_v2_2_5_best_hpo.yaml",
    #    "run_yaml":    "results/test_bcos_res_2_5_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "bcos",
    #},
    #{
    #    "name":        "resnet34",
    #    "gridpath":    "resnet34_default",
    #    "model_yaml":  "training/config/detector/resnet34.yaml",
    #    "run_yaml":    "results/test_res_lime_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "lime",
    #},
    #{
    #    "name":        "resnet34",
    #    "gridpath":    "resnet34_default",
    #    "model_yaml":  "training/config/detector/resnet34.yaml",
    #    "run_yaml":    "results/test_res_gradcam_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "gradcam",
    #},
    #{
    #    "name":        "resnet34",
    #    "gridpath":    "resnet34_default",
    #    "model_yaml":  "training/config/detector/resnet34.yaml",
    #    "run_yaml":    "results/test_res_layergrad_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "layergrad",
    #},
    #{
    #    "name":        "resnet34",
    #    "gridpath":    "resnet34_default",
    #    "model_yaml":  "training/config/detector/resnet34.yaml",
    #    "run_yaml":    "results/test_res_xgrad_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "xgrad",
    #},
    #{
    #    "name":        "resnet34",
    #    "gridpath":    "resnet34_default",
    #    "model_yaml":  "training/config/detector/resnet34.yaml",
    #    "run_yaml":    "results/test_res_grad++_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "grad++",
    #},
    {
        "name":        "xception_bcos_detector",
        "gridpath":    "xception_bcos_detector_2_5",
        "model_yaml":  "training/config/detector/xception_bcos.yaml",
        "run_yaml":    "results/test_bcos_xception_2_5_config.yaml",
        "weights_key": "pretrained",
        "xai":         "bcos",
    },
    {
        "name":        "xception_detector",
        "gridpath":    "xception_default",
        "model_yaml":  "training/config/detector/xception.yaml",
        "run_yaml":    "results/test_xception_layergrad_config.yaml",
        "weights_key": "pretrained",
        "xai":         "layergrad",
    },
    #{
    #    "name":        "xception_detector",
    #    "gridpath":    "xception_default",
    #    "model_yaml":  "training/config/detector/xception.yaml",
    #    "run_yaml":    "results/test_xception_lime_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "lime",
    #},
    #{
    #    "name":        "xception_detector",
    #    "gridpath":    "xception_default",
    #    "model_yaml":  "training/config/detector/xception.yaml",
    #    "run_yaml":    "results/test_xception_gradcam_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "gradcam",
    #},
    #{
    #    "name":        "xception_detector",
    #    "gridpath":    "xception_default",
    #    "model_yaml":  "training/config/detector/xception.yaml",
    #    "run_yaml":    "results/test_xception_grad++_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "grad++",
    #},
    #{
    #    "name":        "xception_detector",
    #    "gridpath":    "xception_default",
    #    "model_yaml":  "training/config/detector/xception.yaml",
    #    "run_yaml":    "results/test_xception_xgrad_config.yaml",
    #    "weights_key": "pretrained",
    #    "xai":         "xgrad",
    #}


    # … add more models here …
]
grid_subpath = "3x3"      # relative inside each results folder
OUTPUT_BASE_DIR = "/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/resultsGPG/comparemodexeption2"
# ────────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Gruppiere Konfigurationen nach eindeutigem gridpath (nicht mehr nur name)
gridpath_to_configs = defaultdict(list)
for cfg in model_configs:
    gridpath_to_configs[cfg["gridpath"]].append(cfg)

unique_gridpaths = list(gridpath_to_configs.keys())
U = len(unique_gridpaths)
K = max(1, N // U)

grid_cache = {}
shared_grid_paths = []

# 2) Für jeden eindeutigen gridpath: Grids einmalig sammeln
for gridpath in unique_gridpaths:
    folder = f"/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/resultsGPG/{gridpath}/{grid_subpath}"


    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    
    pts = sorted(glob.glob(os.path.join(folder, "*.pt")), key=natural_key)
    print(f"→ Found {len(pts)} files.")

    selected = pts[:K]
    shared_grid_paths += selected

    for path in selected:
        if path not in grid_cache:
            grid_cache[path] = torch.load(path, map_location=device)

# 3) Lade die Grids aus dem Cache
shared_grids = [grid_cache[p] for p in shared_grid_paths]

print("\n Shared grids selected for evaluation:")
for i, path in enumerate(shared_grid_paths):
    print(f"  [{i+1:02}] {path}")
print(f"\n→ Total: {len(shared_grid_paths)} grid(s) used across all model configs.")

# 4) Optional: absichern
assert len(shared_grids) == len(shared_grid_paths), "Mismatch in loaded grids"

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
    elif cfg["xai"] in ["gradcam", "xgrad", "grad++", "layergrad"]: evaluator = GradCamEvaluator(model, device, method=cfg["xai"])
    elif cfg["xai"] == "lime":    evaluator = LIMEEvaluator(model, device)
    else: raise RuntimeError(f"Unknown XAI '{cfg['xai']}'")

    print(f"Preparing grids for XAI method: {cfg['xai']}")

    # d) channel handling
    if cfg["xai"] in ("lime", "gradcam", "xgrad", "grad++", "layergrad"):
        grids = []
        for i, g in enumerate(shared_grids):
            print(f"  [LIME-family] Grid {i}: input shape {tuple(g.shape)}")
            if g.ndim == 4:
                g_out = g[:, :3, ...]
            elif g.ndim == 3:
                g_out = g[:3, ...]
            else:
                raise ValueError(f"unexpected grid shape {g.shape}")
            print(f"    → reduced shape: {tuple(g_out.shape)}")
            grids.append(g_out)
    
    elif cfg["xai"] == "bcos":
        grids = []
        for i, g in enumerate(shared_grids):
            print(f"  [BCoS] Grid {i}: input shape {tuple(g.shape)}")
            if g.ndim == 4:
                if g.shape[1] == 3:
                    pad = torch.zeros((g.shape[0], 3, *g.shape[2:]), device=g.device, dtype=g.dtype)
                    g_out = torch.cat([g, pad], dim=1)
                    print(f"    → padded to shape: {tuple(g_out.shape)}")
                else:
                    g_out = g
                    print(f"    → unchanged (already >=6ch): {tuple(g_out.shape)}")
            elif g.ndim == 3:
                if g.shape[0] == 3:
                    pad = torch.zeros((3, *g.shape[1:]), device=g.device, dtype=g.dtype)
                    g_out = torch.cat([g, pad], dim=0)
                    print(f"    → padded to shape: {tuple(g_out.shape)}")
                else:
                    g_out = g
                    print(f"    → unchanged (already >=6ch): {tuple(g_out.shape)}")
            else:
                raise ValueError(f"unexpected grid shape {g.shape}")
            grids.append(g_out)
    
    else:
        grids = shared_grids
        for i, g in enumerate(grids):
            print(f"  [Other XAI] Grid {i}: using as-is → shape {tuple(g.shape)}")

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

    out_dir = os.path.join(OUTPUT_BASE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    all_raw_path = os.path.join(
    out_dir,
    f"results_by_threshold_{cfg['name']}_{cfg['xai']}.pkl"
)
    with open(all_raw_path, "wb") as f:
        pickle.dump(dict(threshold_groups), f)
    print(f" → saved grouped raw results: {all_raw_path}")

#python notebooks/Linus/GridPointingGame/COMPARE_eval.py
