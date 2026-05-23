import os
import sys

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
    """Converts a tensor to a NumPy array."""
    return t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t

class HeatmapEvaluator:
    @staticmethod
    def grad_to_img(img, linear_mapping, alpha_percentile=99.5):
        """Compute an RGBA heatmap from the dynamic linear mapping of BCos models."""
        contribs = (img * linear_mapping).sum(0, keepdim=True)[0]
        logging.debug("Contribution stats: min=%s, max=%s", contribs.min().item(), contribs.max().item())

        rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12)).clamp(0)
        logging.debug("RGB grad stats: min=%s, max=%s", rgb_grad[:3].min().item(), rgb_grad[:3].max().item())

        rgb_grad_np = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))
        alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        alpha = to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
        logging.debug("Alpha stats: min=%s, max=%s", alpha.min(), alpha.max())

        rgb_grad_np = np.concatenate([rgb_grad_np, alpha], axis=0)
        heatmap = rgb_grad_np.transpose((1, 2, 0))
        logging.debug("Final heatmap shape: %s", heatmap.shape)
        return heatmap

    @staticmethod
    def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None):
        """
        Evaluate the heatmap to compute the intensity-weighted accuracy.
        """
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)
        if heatmap_gray.max() > 1.0:
            heatmap_gray = heatmap_gray / 255.0

        intensity_cap = np.percentile(heatmap_gray, top_percentile)
        heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

        original_size = heatmap_gray.shape
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
            pil_img = Image.fromarray((heatmap_gray * 255).astype(np.uint8))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            heatmap_gray = np.array(pil_img).astype(np.float32) / 255.0
        else:
            new_size = original_size

        rows, cols = heatmap_gray.shape
        section_size_row = rows // grid_split
        section_size_col = cols // grid_split
        sections = [heatmap_gray[i * section_size_row:(i + 1) * section_size_row,
                                   j * section_size_col:(j + 1) * section_size_col]
                    for i in range(grid_split) for j in range(grid_split)]
        intensity_sums = [np.sum(section) for section in sections]
        guessed_fake_position = np.argmax(intensity_sums)
        total_positive_intensity = np.sum(heatmap_gray)
        intensity_weighted_accuracy = (intensity_sums[true_fake_pos] / total_positive_intensity
                                       if total_positive_intensity > 0 else 0.0)
        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy

class BCOSEvaluator:
    def __init__(self, model=None, device=None):
        """Initializes the evaluator with a model."""
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        logging.info("Loaded model %s onto %s", self.model.__class__.__name__, self.device)

    def generate_heatmap(self, tensor):
        """Generate heatmap using gradients."""
        img = tensor.to(self.device).requires_grad_(True)
        logging.debug("Input tensor shape: %s", img.shape)
        self.model.zero_grad()
        out = self.model({'image': img})
        logging.debug("Model output: %s", out)
        scalar_out = out['prob'][0]
        scalar_out.backward()
        grad = img.grad[0]
        logging.debug("Gradients: min=%s, max=%s, mean=%s", grad.min().item(), grad.max().item(), grad.mean().item())
        heatmap = HeatmapEvaluator.grad_to_img(img[0], grad, alpha_percentile=100)
        logging.debug("Heatmap: shape=%s, min=%s, max=%s", heatmap.shape, heatmap.min(), heatmap.max())
        return to_numpy(heatmap), out

    def convert_to_numpy(self, tensor):
        """Convert a tensor to an RGB numpy image."""
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return (to_numpy(tensor[:3].permute(1, 2, 0)) * 255).astype(np.uint8)

    def extract_fake_position(self, path):
        """Extract fake image position from filename."""
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            logging.warning("Could not extract fake position from '%s': %s", path, e)
            return -1

    def evaluate(self, tensor_list, path_list, grid_split):
        """Evaluate grid tensors and compute metrics."""
        results = []
        logging.info("Processing %d grids with grid_split=%d.", len(tensor_list), grid_split)
        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            logging.info("Evaluating grid %d from file: %s", idx, path)
            if tensor.shape[1] == 3:
                tensor = torch.cat([tensor, 1.0 - tensor], dim=1)
            heatmap, output = self.generate_heatmap(tensor)
            true_fake_pos = self.extract_fake_position(path)
            guessed_fake_position, intensity_sums, accuracy = HeatmapEvaluator.evaluate_heatmap(
                heatmap, grid_split=grid_split, true_fake_pos=true_fake_pos
            )
            original_image = self.convert_to_numpy(tensor)
            result = {
                "path": path,
                "original_image": original_image,
                "heatmap": heatmap,
                "guessed_fake_position": guessed_fake_position,
                "accuracy": accuracy,
                "true_fake_position": true_fake_pos,
            }
            results.append(result)
            logging.info("For grid %s: true position %d, predicted %d, grid accuracy: %.3f",
                        os.path.basename(path), true_fake_pos, guessed_fake_position, accuracy)
        return results