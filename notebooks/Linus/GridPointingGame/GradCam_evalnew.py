# gradcam
import os
import sys 

# Set up project root and ensure it's in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, XGradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from training.detectors.resnet34_detector import ResnetDetector
#from B_COS_eval import evaluate_heatmap
from training.detectors.xception_detector import XceptionDetector
#from training.detectors.vgg_detector import VGGDetector  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None, threshold = 0.2):
    """Evaluate heatmap; returns guessed cell, cell intensity sums, and accuracy."""
    # only diff to bcos method is the heatmap is recieved in grayscale
    heatmap_intensity = heatmap

    print(f"shape: {heatmap_intensity.shape}")
    # Calculate cell dimensions.
    rows, cols = heatmap_intensity.shape
    sec_rows = rows // grid_split
    sec_cols = cols // grid_split
    # Split into grid cells.
    sections = [heatmap_intensity[i*sec_rows:(i+1)*sec_rows, j*sec_cols:(j+1)*sec_cols]
                for i in range(grid_split) for j in range(grid_split)]
    # Sum intensity in each cell.
    intensity_sums = [np.sum([section>threshold]) for section in sections]
    for i, intensity in enumerate(intensity_sums):
        print("Intensitätssumme für Zelle {}: {}".format(i, intensity))
    guessed_fake_position = np.argmax(intensity_sums)
    total_intensity = np.sum(intensity_sums)
    # Compute accuracy as fraction of total intensity in the true fake cell.
    accuracy = (intensity_sums[true_fake_pos] / total_intensity) if total_intensity > 0 else 0.0
    return guessed_fake_position, intensity_sums, accuracy


# Auto-find the last valid Conv2d layer in the backbone
# Excludes layers with names containing 'adjust' or 'proj'
def find_last_valid_conv_layer(module):
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and all(x not in name for x in ["adjust", "proj"]):
            last_conv = m
    return last_conv

# Wraps the detector so CAM methods receive tensor input
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # Detector expects dict with 'image'
        return self.model({"image": x})["cls"]

class GradCamEvaluator:
    def __init__(self, model, device, method="gradcam"):
        self.model = model.to(device)
        self.device = device
        self.method = method.lower()

        # dynamically find the target layer
        self.target_layer = find_last_valid_conv_layer(self.model.backbone)
        if self.target_layer is None:
            raise ValueError("No valid Conv2d layer found in model backbone.")
        logger.info("Selected target layer for XAI: %s", self.target_layer.__class__.__name__)

        self.target_layer = dict(self.model.backbone.named_modules())["resnet.7.2.conv2"]
        logger.info("Selected target layer for XAI: %s", self.target_layer.__class__.__name__)

        # wrap detector for CAM
        self.wrapped_model = WrappedModel(self.model)

        # select CAM method
        if self.method == "gradcam":
            self.cam = GradCAM(model=self.wrapped_model, target_layers=[self.target_layer])
        elif self.method == "xgrad":
            self.cam = XGradCAM(model=self.wrapped_model, target_layers=[self.target_layer])
        elif self.method == "grad++":
            self.cam = GradCAMPlusPlus(model=self.wrapped_model, target_layers=[self.target_layer])
        elif self.method == "layergrad":
            # Captum's LayerGradCam uses the unwrapped model and direct layer
            self.cam = LayerGradCam(model=self.model, layer=self.target_layer)
        else:
            raise ValueError(f"Unknown CAM method: {self.method}")

    def extract_fake_position(self, path):
        # extracts fake position index from filename
        try:
            return int(os.path.basename(path).split("_fake_")[1].split('.')[0])
        except:
            logger.warning("Could not extract fake position from %s", path)
            return -1

    def convert_to_numpy(self, tensor):
        # convert single image tensor to uint8 numpy HWC
        t = tensor.squeeze(0)
        arr = t.permute(1,2,0).detach().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    def generate_heatmap(self, tensor, true_fake_pos, grid_split):
        # tensor: single image tensor (C,H,W)
        inp = tensor.unsqueeze(0).to(self.device)
        if self.method == "layergrad":
            attributions = self.cam.attribute(inp, target=true_fake_pos)
            grayscale_cam = attributions.squeeze().cpu().detach().numpy()
        else:
            grayscale_cam = self.cam(input_tensor=tensor.unsqueeze(0), targets=[ClassifierOutputTarget(1)])[0]
        # original image for overlay
        img = tensor.cpu().permute(1,2,0).numpy()
        img = (img - img.min())/(img.max()-img.min()+1e-8)
        heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        return grayscale_cam, heatmap

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
        results = []
        for tensor, path in zip(tensor_list, path_list):
            true_pos = self.extract_fake_position(path)
            intensity_map, heatmap = self.generate_heatmap(tensor[0], true_pos, grid_split)
            orig = self.convert_to_numpy(tensor[0])
            # threshold evaluations
            thresholds = [None] + ([i/threshold_steps for i in range(1, threshold_steps+1)] if threshold_steps>0 else [])
            for t in thresholds:
                guessed_pos, intensity_sums, acc = evaluate_heatmap(
                    heatmap=intensity_map, grid_split=grid_split,
                    true_fake_pos=true_pos, threshold=(t or 0)
                )
                results.append({
                    "path": path,
                    "threshold": t,
                    "true_pos": true_pos,
                    "guessed_pos": guessed_pos,
                    "accuracy": acc,
                    "heatmap": heatmap,
                    "original": orig
                })
        return results