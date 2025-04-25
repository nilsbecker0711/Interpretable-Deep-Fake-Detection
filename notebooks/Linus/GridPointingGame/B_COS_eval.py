import os
import sys

# Set up project root and ensure it's in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(t):
    """Convert tensor to numpy array."""
    return t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t

def evaluate_heatmap(heatmap, grid_split=3, true_fake_pos=None):
    """Evaluate heatmap; returns guessed cell, cell intensity sums, and accuracy."""
    # Convert heatmap to grayscale (average first 3 channels).
    heatmap_intensity = heatmap[:,:,-1]
    print(f"shape: {heatmap_intensity.shape}")

    # Calculate cell dimensions.
    rows, cols = heatmap_intensity.shape
    sec_rows = rows // grid_split
    sec_cols = cols // grid_split

    # Split into grid cells.
    sections = [heatmap_intensity[i*sec_rows:(i+1)*sec_rows, j*sec_cols:(j+1)*sec_cols]
                for i in range(grid_split) for j in range(grid_split)]

    # Count of pixels with intensity in each cell.
    intensity_counts = [np.sum(section > background_pixel) for section in sections]
    fake_pred_unweighted = np.argmax(intensity_counts)

    # unweighted prediction 
    total_nonzero_count = float(sum(intensity_counts))
 
    if total_nonzero_count > 0 and 0 <= true_fake_pos < len(intensity_counts):
        unweighted_grid_accuracy = intensity_counts[true_fake_pos] / total_nonzero_count
    
    # Sum intensity in each cell.
    intensity_sums = [np.sum(section) for section in sections]
    for i, intensity in enumerate(intensity_sums):
        print("Intensitätssumme für Zelle {}: {}".format(i, intensity))
    fake_pred_weighted = np.argmax(intensity_sums)
    total_intensity = np.sum(intensity_sums)
    
    # Compute weighted accuracy as fraction of total intensity in the true fake cell.
    weighted_accuracy = (intensity_sums[true_fake_pos] / total_intensity) if total_intensity > 0 else 0.0

    
    return fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy

class BCOSEvaluator:
    def __init__(self, model=None, device=None):
        """Initialize with model and device."""
        self.model = model
        self.device = device

    def generate_heatmap(self, tensor):
        """Generate heatmap via forward and backward passes."""
        # Move tensor to device and enable gradients.
        img = tensor.to(self.device).requires_grad_(True)
        logger.debug("Input tensor shape: %s", img.shape)

        self.model.zero_grad()  # Reset gradients.
        out = self.model({'image': img})  # Forward pass.
        
        # ─── Logit-Check ───────────────────────────────────────────────
        # raw logits before softmax/sigmoid
        logits = out['cls'][0]  
        logger.info(
            "Logits → min: %.4f   mean: %.4f   max: %.4f",
            logits.min().item(), logits.mean().item(), logits.max().item()
        )
        # ────────────────────────────────────────────────────────────────
        
        logger.debug("Model output: %s", out)
        
        scalar_out = out['prob'][0]  # Use first output probability.
        scalar_out.backward()  # Backpropagate.
        grad = img.grad[0]
        logger.debug("Gradients: min=%s, max=%s, mean=%s",
                     grad.min().item(), grad.max().item(), grad.mean().item())
        
        # Get explanation from model's backbone.
        explanation = self.model.backbone.explain(img, idx=1)

        # ────────────────────────────────────────────────────────────────
        # Log dynamic linear weights
        dyn = explanation["dynamic_linear_weights"]
        logger.info(
            "DynWeights → min: %.4f  mean: %.4f  max: %.4f",
            dyn.min().item(), dyn.mean().item(), dyn.max().item()
        )

        # Log contribution map
        cmap = explanation["contribution_map"]
        logger.info(
            "ContribMap → min: %.4f  mean: %.4f  max: %.4f",
            cmap.min().item(), cmap.mean().item(), cmap.max().item()
        )
        # ────────────────────────────────────────────────────────────────


        heatmap = explanation.get("explanation")
        model_prediction = explanation.get("prediction")

        #heatmap = explanation["explanation"][:,:,:].copy()
        
        if heatmap is None:
            logger.error("No heatmap found. Keys: %s", explanation.keys())
            raise ValueError("Heatmap extraction failed.")
        logger.debug("Heatmap: shape=%s, min=%s, max=%s", heatmap.shape, heatmap.min(), heatmap.max())
        return to_numpy(heatmap), out, model_prediction

    def convert_to_numpy(self, tensor):
        """Convert tensor to RGB numpy image."""
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # Remove batch dimension.
        # Permute channels to HWC and scale.
        return (to_numpy(tensor[:3].permute(1, 2, 0)) * 255).astype(np.uint8)

    def extract_fake_position(self, path):
        """Extract fake position from filename."""
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            logger.warning("Could not extract fake position from '%s': %s", path, e)
            return -1

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
        """Evaluate grid tensors and return metrics."""
        results = []
        logger.info("Processing %d grids with grid_split=%d.", len(tensor_list), grid_split)

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)
            # If tensor has 3 channels, add inverse channels.
            if tensor.shape[1] == 3:
                tensor = torch.cat([tensor, 1.0 - tensor], dim=1)
            heatmap, output, model_prediction = self.generate_heatmap(tensor)
            true_fake_pos = self.extract_fake_position(path)
            original_image = self.convert_to_numpy(tensor)

            thresholds = [None]  # No threshold
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]
                
            def apply_threshold(heatmap, threshold):        
                if threshold is None:
                    return heatmap.copy()
                thresholded = heatmap.copy()
                #thresholded[:, :, -1] = (thresholded[:, :, -1] > threshold).astype(np.uint8)

                # extract the alpha channel
                alpha = thresholded[:, :, 3]
                # zero-out anything under threshold
                mask = alpha >= threshold
                alpha_thresholded = alpha.copy()
                alpha_thresholded[~mask] = 0.0
                thresholded[:, :, 3] = alpha_thresholded
                return thresholded

            for t in thresholds:
                logger.info("Evaluating with threshold: %s", t if t is not None else "no threshold")
                thresholded_heatmap = apply_threshold(heatmap, t)

                fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy = evaluate_heatmap(thresholded_heatmap, grid_split=grid_split, true_fake_pos=true_fake_pos)

                result = {
                    "threshold": t if t is not None else 0,                    
                    "path": path,
                    "original_image": original_image,
                    "heatmap": thresholded_heatmap,
                    "weighted_guessed_fake_position": fake_pred_weighted,
                    "unweighted_guess_fake_position": fake_pred_unweighted,                    
                    "weighted_localization_score": weighted_accuracy,
                    "unweighted_localization_score": unweighted_accuracy,
                    "true_fake_position": true_fake_pos,
                    "model_prediction": model_prediction
                }
                
                results.append(result)
                
                logger.info("Threshold %s | %s: true pos %d, predicted (weighted) %d, accuracy (weighted): %.3f | predicted (unweighted) %d, accuracy (unweighted): %.3f",
                            str(t), os.path.basename(path), true_fake_pos, fake_pred_weighted, grid_accuracy, fake_pred_unweighted, unweighted_grid_accuracy)
                
        return results