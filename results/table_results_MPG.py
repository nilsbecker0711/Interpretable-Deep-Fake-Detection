import os
import pickle
import numpy as np

# ─────────── CONFIG ───────────
models = [
    #"resnet34_bcos_v2_test_MPG_bcos_1_25",
    #"resnet34_bcos_v2_test_MPG_gradcam",
    #"resnet34_bcos_v2_test_MPG_grad++",
    #"resnet34_bcos_v2_test_MPG_xgrad",
    #"resnet34_bcos_v2_test_MPG_layergrad",
    #"resnet34_bcos_v2_test_MPG_bcos_1_75",
    #"resnet34_bcos_v2_test_MPG_bcos_2",
    #"resnet34_bcos_v2_test_MPG_bcos_2_5",
    #"resnet34_test_MPG_lime",
    #"resnet34_test_MPG_gradcam",
    #"resnet34_test_MPG_xgrad",
    #"resnet34_test_MPG_grad++",
    #"resnet34_test_MPG_layergrad",
    
    #"vit_bcos_test_MPG_bcos_1_25",
    #"vit_bcos_test_MPG_bcos_1_75",
    #"vit_bcos_test_MPG_bcos_2",
    #"vit_bcos_test_MPG_bcos_2_5",
    #"vit_test_MPG_lime",
    #"vit_test_MPG_gradcam",
    #"vit_test_MPG_xgrad",
    #"vit_test_MPG_grad++",
    #"vit_test_MPG_layergrad",
    
    "xception_test_MPG_lime",
    #"xception_test_MPG_gradcam",
    #"xception_test_MPG_xgrad",
    #"xception_test_MPG_grad++",
    #"xception_test_MPG_layergrad",
    #"xception_bcos_detector_test_MPG_bcos_2_5"

    # ggf. weitere Modelle hier
]
results_base = "/pfs/work9/workspace/scratch/ma_tischuet-team_project_explainable_deepfakes/resultsMPG/"
#grid_subpath  = "MaskPointingGame"
threshold_key = 0  # unser „no threshold“, wird im Dict als 0 gespeichert
# ───────────────────────────────

print("| Model                                           | Weighted avg | Unweighted avg |")
print("|:------------------------------------------------|------------:|---------------:|")

for m in models:
    pkl = os.path.join(results_base, m, "MaskPointingGame/results_by_threshold.pkl")
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