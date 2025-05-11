import os
import pickle
import numpy as np

# ─────────── CONFIG ───────────
models = [
    "resnet34_test_res_layergrad_config",
    #"xception_bcos_detector_test_bcos_xception_2_5_config",
    #"xception_bcos_v2_test_bcos_xception_2_config",
    #"xception_bcos_v2_test_bcos_xception_1_25_config",
    #"xception_test_xception_lime_config",
    #"xception_test_xception_xgrad_config",
    "xception_test_xception_layergrad_config",
    #"xception_test_xception_grad++_config",
    #"xception_test_xception_gradcam_config"
    # ggf. weitere Modelle hier
    "resnet34_bcos_v2_test_bcos_res_gradcam_config",
    "resnet34_bcos_v2_test_bcos_res_grad++_config",
    "resnet34_bcos_v2_test_bcos_res_layergrad_config",
    "resnet34_bcos_v2_test_bcos_res_xgrad_config",
    "resnet34_bcos_v2_test_res_gradcam_config"
]
results_base = "/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/resultsGPG"
grid_subpath  = "3x3"
threshold_key = 0  # unser „no threshold“, wird im Dict als 0 gespeichert
# ───────────────────────────────

print("| Model                                           | Weighted avg | Unweighted avg |")
print("|:------------------------------------------------|------------:|---------------:|")

for m in models:
    pkl = os.path.join(results_base, m, grid_subpath, "results_by_threshold.pkl")
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    entries = res.get(threshold_key, [])
    if not entries:
        print(f"| {m:48} |      n/a    |       n/a     |")
        continue

    w_scores = [e["weighted_localization_score"]   for e in entries]
    u_scores = [e["unweighted_localization_score"] for e in entries]
    w_mean = np.mean(w_scores)
    u_mean = np.mean(u_scores)

    print(f"| {m:48} | {w_mean:12.4f} | {u_mean:14.4f} |")
    print 