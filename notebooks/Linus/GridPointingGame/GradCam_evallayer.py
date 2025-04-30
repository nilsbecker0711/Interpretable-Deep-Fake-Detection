import os
import sys

# Set up project root and ensure it's in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm  # if not used, you can remove
from pytorch_grad_cam import GradCAM, XGradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import LayerGradCam

from training.detectors.resnet34_detector import ResnetDetector
from training.detectors.xception_detector import XceptionDetector
# from training.detectors.vgg_detector import VGGDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_heatmap(heatmap, grid_split=3, true_fake_pos=None, background_pixel=0):
    """Evaluate heatmap; returns guessed cell, cell intensity sums, and accuracy."""
    heatmap_intensity = heatmap
    rows, cols = heatmap_intensity.shape
    sec_rows = rows // grid_split
    sec_cols = cols // grid_split

    sections = [
        heatmap_intensity[i*sec_rows:(i+1)*sec_rows, j*sec_cols:(j+1)*sec_cols]
        for i in range(grid_split) for j in range(grid_split)
    ]

    # unweighted: count of “on” pixels
    intensity_counts = [np.sum(sec > background_pixel) for sec in sections]
    fake_pred_unweighted = int(np.argmax(intensity_counts))
    total_nonzero = float(sum(intensity_counts))
    unweighted_accuracy = (
        intensity_counts[true_fake_pos] / total_nonzero
        if total_nonzero > 0 and 0 <= true_fake_pos < len(intensity_counts) else 0.0
    )

    # weighted: sum of intensities
    intensity_sums = [float(np.sum(sec)) for sec in sections]
    fake_pred_weighted = int(np.argmax(intensity_sums))
    total_intensity = float(sum(intensity_sums))
    weighted_accuracy = (
        intensity_sums[true_fake_pos] / total_intensity
        if total_intensity > 0 and 0 <= true_fake_pos < len(intensity_sums) else 0.0
    )

    return fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy


def find_last_valid_conv_layer(module):
    """Return the last nn.Conv2d (no 'adjust'/'proj') in module."""
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and all(x not in name for x in ("adjust", "proj")):
            last_conv = m
    return last_conv


def find_all_valid_conv_layers(module):
    """Return list of (name, nn.Conv2d) excluding 'adjust'/'proj'."""
    layers = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and all(x not in name for x in ("adjust", "proj")):
            layers.append((name, m))
    return layers


class WrappedModel(nn.Module):
    """Wrap detector so CAM APIs see a simple forward(x)->logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # detectors expect a dict with key 'image'
        return self.model({"image": x})["cls"]


class GradCamEvaluator:
    def __init__(self, model, device, method="gradcam"):
        self.model = model.to(device)
        self.device = device
        self.method = method.lower()

        # pick CAM target layer (fallback if no auto-search)
        self.target_layer = find_last_valid_conv_layer(self.model.backbone)
        if self.target_layer is None:
            raise ValueError("No valid Conv2d layer found in model backbone.")
        logger.info("Selected target layer for XAI: %s",
                    self.target_layer.__class__.__name__)

        # for layergrad only: we'll later override this if auto-search picks another
        self.best_layer_name = None
        self.best_layer_idx  = None

        self.wrapped_model = WrappedModel(self.model)

        # choose CAM backend
        if self.method == "gradcam":
            self.cam = GradCAM(model=self.wrapped_model,
                               target_layers=[self.target_layer])
        elif self.method == "xgrad":
            self.cam = XGradCAM(model=self.wrapped_model,
                                target_layers=[self.target_layer])
        elif self.method == "grad++":
            self.cam = GradCAMPlusPlus(model=self.wrapped_model,
                                       target_layers=[self.target_layer])
        elif self.method == "layergrad":
            self.cam = LayerGradCam(self.wrapped_model, self.target_layer)
        else:
            raise ValueError(f"Unknown CAM method: {self.method}")

    def extract_fake_position(self, path):
        try:
            return int(os.path.basename(path).split("_fake_")[1].split('_conf_')[0])
        except:
            logger.warning("Could not extract fake position from %s", path)
            return -1

    def convert_to_numpy(self, tensor):
        """Convert [1,C,H,W] float in [0,1] to uint8 H×W×3."""
        t = tensor.squeeze(0).detach().cpu()
        img = t.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)

    def generate_heatmap(self, tensor):
        # get the raw 2D map
        if self.method == "layergrad":
            inp = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            cam_map = self.cam.attribute(inp, target=1)
            grayscale_cam = cam_map.squeeze().detach().cpu().numpy()
            # ReLU only for layergrad
            grayscale_cam = np.maximum(grayscale_cam, 0)
            if grayscale_cam.max() > 0:
                grayscale_cam /= (grayscale_cam.max() + 1e-8)
        else:
            # pytorch_grad_cam backends already apply ReLU internally
            grayscale_cam = self.cam(
                input_tensor=tensor.unsqueeze(0),
                targets=[ClassifierOutputTarget(1)]
            )[0]

        # normalize the RGB for overlay
        rgb_img = tensor.detach().cpu().permute(1,2,0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

        # upsample when needed
        if self.method == "layergrad":
            h, w, _ = rgb_img.shape
            grayscale_cam = cv2.resize(grayscale_cam, (w, h),
                                      interpolation=cv2.INTER_LINEAR)

        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return grayscale_cam, rgb_img, heatmap

    def evaluate(self, tensor_list, path_list, grid_split,
                 threshold_steps=0, auto_search_k=5):
        """
        Runs CAM on each tensor, thresholds the map, and computes localization.
        If method=="layergrad", auto-search best conv-layer over the first K grids.
        """
        # ─── Auto-search best layer for LayerGradCam ─────────────────────
        if self.method == "layergrad" and not getattr(self, "_searched_layer", False):
            self._searched_layer = True
            all_convs = find_all_valid_conv_layers(self.model.backbone)
            best_score, best_name, best_layer, best_idx = -1, None, None, None
            sample_n = min(auto_search_k, len(tensor_list))

            for idx, (name, conv) in enumerate(all_convs):
                cam_try = LayerGradCam(self.wrapped_model, conv)
                scores = []
                for tensor_grid, path in zip(tensor_list[:sample_n], path_list[:sample_n]):
                    img = tensor_grid[0]                             # [C,H,W]
                    inp = img.unsqueeze(0).to(self.device)           # [1,C,H,W]
                    inp.requires_grad_(True)
                    attr = cam_try.attribute(inp, target=1)
                    gcam = attr.squeeze(0).mean(dim=0).cpu().detach().numpy()
                    gcam = np.maximum(gcam, 0)
                    if gcam.max() > 0:
                        gcam /= (gcam.max() + 1e-8)
                    _, _, wa, _, _ = evaluate_heatmap(
                        heatmap=gcam,
                        grid_split=grid_split,
                        true_fake_pos=self.extract_fake_position(path)
                    )
                    scores.append(wa)

                avg = float(np.mean(scores))
                logger.info(f"[AutoSearch] Layer {idx:2d} ({name}): mean weighted {avg:.3f}")
                if avg > best_score:
                    best_score, best_name, best_layer, best_idx = avg, name, conv, idx

            if best_name is None:
                logger.warning("[AutoSearch] couldn’t pick a best layer (all scores nan); keeping initial layer %s",
                               self.target_layer.__class__.__name__)
            else:
                # store and switch to the best layer
                self.best_layer_name = best_name
                self.best_layer_idx  = best_idx
                self.target_layer    = best_layer
                self.cam             = LayerGradCam(self.wrapped_model, self.target_layer)
                logger.info(f"[AutoSearch] Picked layer “{best_name}” (index {best_idx}) with score {best_score:.3f}")

        # ─── Full dataset evaluation ────────────────────────────────────
        results = []
        logger.info("Processing %d grids with grid_split=%d using CAM method=%s",
                    len(tensor_list), grid_split, self.method)

        for idx, (tensor_grid, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)
            true_fake_pos = self.extract_fake_position(path)

            intensity_map, norm_img, _ = self.generate_heatmap(tensor_grid[0])
            original_image = self.convert_to_numpy(tensor_grid[0])
            model_prediction = 1  # still hard-coded

            thresholds = [None]
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps+1)]

            for t in thresholds:
                t_desc = "none" if t is None else f"{t:.3f}"
                logger.info("  Threshold = %s", t_desc)

                mask = intensity_map if t is None else np.where(intensity_map < t, 0.0, intensity_map)
                thresholded_overlay = show_cam_on_image(norm_img, mask, use_rgb=True)

                fake_pred_weighted, intensity_sums, weighted_accuracy, \
                fake_pred_unweighted, unweighted_accuracy = evaluate_heatmap(
                    heatmap=mask,
                    grid_split=grid_split,
                    true_fake_pos=true_fake_pos
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
                    "unweighted_localization_score": unweighted_accuracy,
                    "model_prediction": model_prediction
                }

                # Now log the *chosen* layer name (if any)
                logger.info(
                    "Method %s | Layer %s | Threshold %s | %s: "
                    "true pos %d, predicted (weighted) %d, accuracy (weighted): %.3f | "
                    "predicted (unweighted) %d, accuracy (unweighted): %.3f",
                    self.method,
                    getattr(self, "best_layer_name", "n/a"),
                    t_desc,
                    os.path.basename(path),
                    true_fake_pos,
                    fake_pred_weighted,
                    weighted_accuracy,
                    fake_pred_unweighted,
                    unweighted_accuracy
                )

                results.append(result)

        return results
