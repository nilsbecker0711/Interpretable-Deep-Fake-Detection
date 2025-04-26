# gradcam and variations 
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
from captum.attr import LayerGradCam

from training.detectors.resnet34_detector import ResnetDetector
#from B_COS_eval import evaluate_heatmap
from training.detectors.xception_detector import XceptionDetector
#from training.detectors.vgg_detector import VGGDetector  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None):
    """Evaluate heatmap; returns guessed cell, cell intensity sums, and accuracy."""
    # heatmap is recieved in grayscale
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
    intensity_sums = [np.sum([section>0]) for section in sections]
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

        #if auto doesnt work 
        #self.target_layer = dict(self.model.backbone.named_modules())["resnet.7.2.conv2"]
        #logger.info("Selected target layer for XAI: %s", self.target_layer.__class__.__name__)

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
           # Captum LayerGradCam expects positional args: (forward_func, layer)
            self.cam = LayerGradCam(self.wrapped_model, self.target_layer)
        else:
            raise ValueError(f"Unknown CAM method: {self.method}")

    def extract_fake_position(self, path):
        try:
            return int(os.path.basename(path).split("_fake_")[1].split('.')[0])
        except:
            logger.warning("Could not extract fake position from %s", path)
            return -1

    def convert_to_numpy(self, tensor):
        """Convert a [C,H,W] float tensor in [0..1] to HxW x 3 uint8."""
        tensor = tensor.squeeze(0)
        np_img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
        return np_img

    def generate_heatmap(self, tensor, path, grid_split, true_fake_pos):
        # First, get a 2D “grayscale_cam” of shape [H,W]:
        if self.method == "layergrad":
            # Captum expects requires_grad on the inputs
            inp = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            # target=1 because you’re hard-coding fake=1
            attributions = self.cam.attribute(inp, target=1)  # -> [1, C, H, W]
            # collapse channels by mean (you could also use abs()+sum)
            grayscale_cam = attributions.squeeze(0).mean(dim=0).detach().cpu().numpy()
            # rectify and normalize to [0..1]
            grayscale_cam = np.maximum(grayscale_cam, 0)
            if grayscale_cam.max() > 0:
                 grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
        else:
             # PyTorch-Grad-CAM API
            grayscale_cam = self.cam(
                input_tensor=tensor.unsqueeze(0),
                targets=[ClassifierOutputTarget(1)]
             )[0]

    # build overlay
    # rgb = normalized float image (shape H×W×3, values in [0..1]) 
        rgb_img = tensor.cpu().permute(1,2,0).numpy()
        rgb_img = (rgb_img - rgb_img.min())/(rgb_img.max()-rgb_img.min()+1e-8)
        heatmap = show_cam_on_image(rgb_img, cam_map, use_rgb=True)

        return grayscale_cam, rgb_img, heatmap

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
        """
        Runs CAM on each tensor, thresholds the resulting map at various levels,
        and computes guessed fake‐cell and accuracy for each threshold.
        """
        results = []
        logger.info(
            "Processing %d grids with grid_split=%d using CAM method=%s",
            len(tensor_list), grid_split, self.method
        )

        for idx, (tensor_grid, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)

            true_fake_pos = self.extract_fake_position(path)

            # 1) generate raw intensity map and the normalized image
            intensity_map, norm_img, _ = self.generate_heatmap(
                tensor_grid[0], path, grid_split, true_fake_pos
            )
            original_image   = self.convert_to_numpy(tensor_grid[0])
            model_prediction = 1  # still hard‐coded 

        # 2) build list of thresholds to try
            thresholds = [None]
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]

        # 3) loop over thresholds
            for t in thresholds:
                t_desc = "none" if t is None else f"{t:.3f}"
                logger.info("  Threshold = %s", t_desc)

            # apply threshold in one line
                if t is None:
                    mask = intensity_map
                else:
                    mask = np.where(intensity_map < t, 0.0, intensity_map)

            # re‐overlay the (float) mask on the normalized [0..1] image
                thresholded_overlay = show_cam_on_image(norm_img, mask, use_rgb=True)

            # compute guessed cell and accuracy
                guessed_pos, intensity_sums, acc = evaluate_heatmap(
                    heatmap=mask,
                    grid_split=grid_split,
                    true_fake_pos=true_fake_pos,
                )

            # collect result
                result = {
                    "threshold": t,
                    "path": path,
                    "original_image": original_image,
                    "heatmap": thresholded_overlay,
                    "guessed_fake_position": guessed_pos,
                    "true_fake_position": true_fake_pos,
                    "accuracy": acc,
                    "model_prediction": model_prediction,
                }
                logger.info(
                    "    Result: true pos=%d, guessed=%d, accuracy=%.3f",
                    true_fake_pos, guessed_pos, acc
                )
                results.append(result)

        return results
 
#    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
 #       results = []
  #      for tensor, path in zip(tensor_list, path_list):
   #         true_pos = self.extract_fake_position(path)
    #        intensity_map, heatmap = self.generate_heatmap(tensor[0], true_pos, grid_split)
     #       orig = self.convert_to_numpy(tensor[0])
      #     thresholds = [None] + ([i/threshold_steps for i in range(1, threshold_steps+1)] if threshold_steps>0 else [])
       #     for t in thresholds:
        #        guessed_pos, intensity_sums, acc = evaluate_heatmap(
         #           heatmap=intensity_map, grid_split=grid_split,
          #          true_fake_pos=true_pos, threshold=(t or 0)
           #     )
            #    results.append({
             #       "path": path,
              #      "threshold": t,
               #     "true_pos": true_pos,
                #    "guessed_pos": guessed_pos,
                 #   "accuracy": acc,
                  #  "heatmap": heatmap,
                   # "original": orig
#                })
 #       return results