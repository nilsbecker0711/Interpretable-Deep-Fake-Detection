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

from .xai_common import smooth_map, normalize_max

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(t):
    """Convert tensor to numpy array."""
    return t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t

def evaluate_heatmap(heatmap, grid_split=3, true_fake_pos=None, background_pixel=0):
    """Score a 2D attribution map on the grid pointing game."""
    heatmap_intensity = heatmap

        # needs to be defined
    unweighted_accuracy = 0.0
    weighted_accuracy   = 0.0

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
    intensity_sums = [np.sum(section) for section in sections]
    #for i, intensity in enumerate(intensity_sums):
        #print("Intensitätssumme für Zelle {}: {}".format(i, intensity))
    fake_pred_weighted = np.argmax(intensity_sums)
    total_intensity = np.sum(intensity_sums)
    
    # Compute weighted accuracy as fraction of total intensity in the true fake cell.
    weighted_accuracy = (intensity_sums[true_fake_pos] / total_intensity) if total_intensity > 0 else 0.0

    
    return fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy

class BCOSEvaluator:
    def __init__(self, model=None, device=None, smooth=15):
        """Initialize with model, device, and the shared smoothing kernel size."""
        self.model = model
        self.device = device
        self.smooth = smooth

    def generate_heatmap(self, tensor):
        """Generate the B-cos attribution map via explain() in explanation_mode.

        Returns (scored_map, visualization, explanation_dict, model_prediction):
          scored_map    -- 2D [H,W] float map in [0,1]: the POSITIVE-clamped
                           contribution map (x * dynamic-linear grad, summed over
                           channels), smoothed per the original localisation
                           protocol. This is the object all scores are computed on.
          visualization -- the RGBA gradient_to_image rendering, for plots ONLY.
                           Its alpha channel is a percentile-clipped |grad| image
                           and must never be used for scoring.
        """
        img = tensor.to(self.device).requires_grad_(True)
        logger.debug("Input tensor shape: %s", img.shape)

        # explain() handles its own forward+backward inside explanation_mode,
        # ensuring dynamic_scaling is detached so img.grad is the true B-cos
        # linear mapping — not a non-linear gradient.
        explanation = self.model.backbone.explain(img, idx=1)

        contribs = explanation.get("contribution_map")
        model_prediction = explanation.get("prediction")
        if contribs is None:
            logger.error("No contribution map found. Keys: %s", explanation.keys())
            raise ValueError("Contribution map extraction failed.")

        # Original protocol order: smooth the SIGNED map, then keep positive
        # attributions only, then rescale to [0,1] for the threshold sweep
        # (the weighted cell/mask fraction itself is scale-invariant).
        scored_map = smooth_map(contribs[0], self.smooth)
        scored_map = np.clip(scored_map, 0, None)
        scored_map = normalize_max(scored_map)

        visualization = to_numpy(explanation.get("explanation"))
        logger.debug("Scored map: shape=%s, max=%s", scored_map.shape, scored_map.max())
        return scored_map, visualization, explanation, model_prediction

    def convert_to_numpy(self, tensor):
        """Convert a torch tensor to a uint8 RGB NumPy image (H x W x 3)."""
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # remove batch dim
    
        tensor = tensor[:3]  # keep only first 3 channels
        tensor = tensor.detach().cpu()
    
        # Normalize to [0, 1]
        tensor = tensor - tensor.min()
        if tensor.max() > 0:
            tensor = tensor / tensor.max()
    
        # Convert to NumPy HWC and scale to [0, 255]
        np_img = tensor.permute(1, 2, 0).numpy()
        return (np_img * 255).clip(0, 255).astype(np.uint8)

    def extract_fake_position(self, path):
        """Extract fake position from filename."""
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('_conf_')[0])
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
            scored_map, visualization, output, model_prediction = self.generate_heatmap(tensor)
            true_fake_pos = self.extract_fake_position(path)
            original_image = self.convert_to_numpy(tensor)

            thresholds = [None]  # No threshold
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]

            for t in thresholds:
                # threshold the 2D positive-contribution map (same semantics as CAM path)
                thresholded_map = scored_map.copy() if t is None else np.where(scored_map < t, 0.0, scored_map)

                fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy = evaluate_heatmap(thresholded_map, grid_split=grid_split, true_fake_pos=true_fake_pos)

                result = {
                    "threshold": t if t is not None else 0,
                    "path": path,
                    "original_image": original_image,
                    "heatmap": thresholded_map,
                    "visualization": visualization,
                    "weighted_guessed_fake_position": fake_pred_weighted,
                    "unweighted_guess_fake_position": fake_pred_unweighted,
                    "weighted_localization_score": weighted_accuracy,
                    "unweighted_localization_score": unweighted_accuracy,
                    "true_fake_position": true_fake_pos,
                    "model_prediction": model_prediction
                }

                results.append(result)
                
                #logger.info("Threshold %s | %s: true pos %d, predicted (weighted) %d, accuracy (weighted): %.3f | predicted (unweighted) %d, accuracy (unweighted): %.3f",
                            #str(t), os.path.basename(path), true_fake_pos, fake_pred_weighted, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy)
                
        return results