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
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(t):
    """Convert tensor to numpy array."""
    return t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t

def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None):
    """Evaluate heatmap; returns guessed cell, cell intensity sums, and accuracy."""
    # Convert heatmap to grayscale (average first 3 channels).
    heatmap_intensity = heatmap[:,:,-1]
    #if heatmap_gray.max() > 1.0:  # Scale to [0,1] if in 0-255 range.
        #heatmap_gray /= 255.0
    
    # Clip extreme values using the top percentile.
    #intensity_cap = np.percentile(heatmap_gray, top_percentile)
    #heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

    # Resize if dimensions are not evenly divisible by grid_split.
    #original_size = heatmap_gray.shape
    #if original_size[1] % grid_split or original_size[0] % grid_split:
        #new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
        #pil_img = Image.fromarray((heatmap_gray * 255).astype(np.uint8))
        #pil_img = pil_img.resize(new_size, Image.LANCZOS)
        #heatmap_gray = np.array(pil_img).astype(np.float32) / 255.0

    print(f"shape: {heatmap_intensity.shape}")
    # Calculate cell dimensions.
    rows, cols = heatmap_intensity.shape
    sec_rows = rows // grid_split
    sec_cols = cols // grid_split
    # Split into grid cells.
    sections = [heatmap_intensity[i*sec_rows:(i+1)*sec_rows, j*sec_cols:(j+1)*sec_cols]
                for i in range(grid_split) for j in range(grid_split)]
    # Sum intensity in each cell.
    intensity_sums = [np.sum(section) for section in sections]
    for i, intensity in enumerate(intensity_sums):
        print("Intensitätssumme für Zelle {}: {}".format(i, intensity))
    guessed_fake_position = np.argmax(intensity_sums)
    total_intensity = np.sum(intensity_sums)
    # Compute accuracy as fraction of total intensity in the true fake cell.
    accuracy = (intensity_sums[true_fake_pos] / total_intensity) if total_intensity > 0 else 0.0
    return guessed_fake_position, intensity_sums, accuracy

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
        logger.debug("Model output: %s", out)
        
        scalar_out = out['prob'][0]  # Use first output probability.
        scalar_out.backward()  # Backpropagate.
        grad = img.grad[0]
        logger.debug("Gradients: min=%s, max=%s, mean=%s",
                     grad.min().item(), grad.max().item(), grad.mean().item())
        
        # Get explanation from model's backbone.
        explanation = self.model.backbone.explain(img, idx=1)
        heatmap = explanation.get("explanation")
        model_prediction = explanation.get("prediction")

        heatmap = explanation["explanation"][:,:,:].copy()
        heatmap[:,:,-1] = (heatmap[:,:,-1] > 0.5).astype(np.uint8)
        
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

    def evaluate(self, tensor_list, path_list, grid_split):
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
            guessed_fake_position, intensity_sums, acc = evaluate_heatmap(
                heatmap, grid_split=grid_split, true_fake_pos=true_fake_pos
            )
            original_image = self.convert_to_numpy(tensor)
            result = {
                "path": path,
                "original_image": original_image,
                "heatmap": heatmap,
                "guessed_fake_position": guessed_fake_position,
                "accuracy": acc,
                "true_fake_position": true_fake_pos,
                "model_prediction": model_prediction
            }
            results.append(result)
            logger.info("For grid %s: true pos %d, predicted %d, accuracy: %.3f",
                        os.path.basename(path), true_fake_pos, guessed_fake_position, acc)
        return results