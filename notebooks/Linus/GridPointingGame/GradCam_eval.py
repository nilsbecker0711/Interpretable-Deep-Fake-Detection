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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from training.detectors.resnet34_detector import ResnetDetector
#from B_COS_eval import evaluate_heatmap
#from training.detectors.xception_detector import XceptionDetector  # if needed later 
#from training.detectors.vgg_detector import VGGDetector  # if needed later

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None, background_pixel=0):
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

    # unweighted prediction 
    # Count of pixels with intensity in each cell.
    intensity_counts = [np.sum(section > background_pixel) for section in sections]
    fake_pred_unweighted = np.argmax(intensity_counts)

    total_nonzero_count = float(sum(intensity_counts))
 
    if total_nonzero_count > 0 and 0 <= true_fake_pos < len(intensity_counts):
        unweighted_accuracy = intensity_counts[true_fake_pos] / total_nonzero_count

    # weighted prediction 
    # Sum intensity in each cell.
    intensity_sums = [np.sum([section>0]) for section in sections]
    for i, intensity in enumerate(intensity_sums):
        print("Intensitätssumme für Zelle {}: {}".format(i, intensity))
    fake_pred_weighted = np.argmax(intensity_sums)
    total_intensity = np.sum(intensity_sums)
    # Compute weighted_accuracy as fraction of total intensity in the true fake cell.
    weighted_accuracy = (intensity_sums[true_fake_pos] / total_intensity) if total_intensity > 0 else 0.0


    return fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"image": x})["cls"]

class GradCamEvaluator:
    def __init__(self, model, device, model_name="resnet"):
        self.model = model
        self.device = device
        self.model_name = model_name


        if model_name == "resnet":
            self.target_layers = [self.model.backbone.resnet[-1][-1].conv2]
        elif model_name == "xception":
            self.target_layers = [self.model.backbone.conv4.pointwise]
        elif model_name == "vgg":
            self.target_layers = [self.model.backbone.features[-1]]
        else:
            raise ValueError("Unsupported model name for Grad-CAM")
        #problem is likely here - may have to define a custom forward function
        wrapped_model = WrappedModel(self.model)
        
        self.cam = GradCAM(model=wrapped_model, target_layers=self.target_layers)

    def extract_fake_position(self, path):
        try:
            return int(os.path.basename(path).split("_fake_")[1].split(".")[0])
        except Exception as e:
            logger.warning("Could not extract fake position from %s: %s", path, e)
            return -1

    def convert_to_numpy(self, tensor):
        tensor = tensor.squeeze(0)
        return (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    #input tensor was tensor.unsqueeze(0)
    def generate_heatmap(self, tensor, path, grid_split, true_fake_pos):
        #grayscale cam is a 2d intensity map
        grayscale_cam = self.cam(input_tensor=tensor.unsqueeze(0),
                                 targets=[ClassifierOutputTarget(1)])[0]
        #print(f"grayscale cam {grayscale_cam}: shape {grayscale_cam.shape} : max {grayscale_cam.max()} : sum {grayscale_cam.sum()}")
        rgb_img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        #print(f"rgb image: {rgb_img}")
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        #intensity_sum = np.sum(grayscale_cam, axis=(0, 1))
        #guessed_fake_pos = int(np.argmax(intensity_sum)) % grid_split

        return grayscale_cam, rgb_img, heatmap #, guessed_fake_pos, intensity_sum

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
        results = []
        logger.info("Processing %d grids with grid_split=%d", len(tensor_list), grid_split)

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)

            true_fake_pos = self.extract_fake_position(path)

            # Grad-CAM heatmap
            intensity_map, rgb_img, heatmap = self.generate_heatmap(tensor[0], path, grid_split, true_fake_pos)
            original_image = self.convert_to_numpy(tensor[0])
            model_prediction = 1  # hardcoded as before

            thresholds = [None]
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]

            for t in thresholds:
                logger.info("Threshold = %s", t if t is not None else "no threshold")
                # 1) build a binary mask
                if t is None:
                    mask = intensity_map
                else:
                    mask = np.copy(intensity_map)
                    mask[intensity_map < t] = 0.0
            
                # 2) re-draw overlay
                #    show_cam_on_image expects a float [0..1] image + float mask [0..1]
                thresholded_overlay = show_cam_on_image(
                    rgb_img,          # your normalized [H,W,3] original
                    mask,             # now thresholded
                    use_rgb=True
                )
            
                # 3) compute cell guesses & accuracy as before
                fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy = evaluate_heatmap(
                    heatmap=mask,      # pass the 2D mask for counting
                    grid_split=grid_split,
                    true_fake_pos=true_fake_pos,
                )

                result = {
                    "threshold": t if t is not None else 0,
                    "path": path,
                    "original_image": original_image,
                    "heatmap": thresholded_overlay,
                    "weighted_guessed_fake_position": fake_pred_weighted,
                    "unweighted_guess_fake_position": fake_pred_unweighted,
                    "true_fake_position": true_fake_pos,
                    "weighted_localization_score": weighted_accuracy,
                    "unweighted_localization_score": fake_pred_unweighted,
                    "model_prediction": model_prediction
                }


                logger.info("Threshold %s | %s: true pos %d, predicted (weighted) %d, accuracy (weighted): %.3f | predicted (unweighted) %d, accuracy (unweighted): %.3f",
                            str(t), os.path.basename(path), true_fake_pos, fake_pred_weighted, weighted_accuracy, fake_pred_unweighted, unweighted_grid_accuracy)

                results.append(result)

        return results