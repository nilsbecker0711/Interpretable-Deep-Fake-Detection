import os
import sys
import numpy as np
import torch
from PIL import Image

from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR


def to_numpy(t):
    """
    Converts a tensor to a numpy array if it is not already one.
    """
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()

class HeatmapEvaluator:
    @staticmethod
    def grad_to_img(img, linear_mapping, alpha_percentile=99.5):
        """
        Compute an RGBA heatmap from the dynamic linear mapping of BCos models.
        """
        # (1) Compute overall contribution for each pixel.
        contribs = (img * linear_mapping).sum(0, keepdim=True)[0]
        print(f"[DEBUG] Contribution stats: min={contribs.min().item()}, max={contribs.max().item()}")

        # (2) Normalize the linear mapping so values fall between 0 and 1.
        rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12)).clamp(0)
        print(f"[DEBUG] RGB grad (first 3 channels) stats: min={rgb_grad[:3].min().item()}, max={rgb_grad[:3].max().item()}")

        # (3) Normalize the first three channels by the sum of positive and negative parts.
        rgb_grad_np = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))
        
        # (4) Compute the opacity (alpha) channel from the mapping.
        alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        alpha = to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
        print(f"[DEBUG] Alpha channel stats: min={alpha.min()}, max={alpha.max()}")

        # (5) Concatenate the normalized RGB mapping with the alpha channel.
        rgb_grad_np = np.concatenate([rgb_grad_np, alpha], axis=0)
        heatmap = rgb_grad_np.transpose((1, 2, 0))
        print(f"[DEBUG] Final heatmap shape: {heatmap.shape}")
        return heatmap

    @staticmethod
    def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, true_fake_pos=None):
        """
        Evaluates the heatmap using intensity-weighted accuracy based on positive attributions.
        """
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)
        if heatmap_gray.max() > 1.0:
            heatmap_gray = heatmap_gray / 255.0

        intensity_cap = np.percentile(heatmap_gray, top_percentile)
        heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

        original_size = heatmap_gray.shape  # (H, W)
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
        else:
            new_size = original_size

        pil_img = Image.fromarray((heatmap_gray * 255).astype(np.uint8))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        heatmap_gray = np.array(pil_img).astype(np.float32) / 255.0

        rows, cols = heatmap_gray.shape
        section_size_row = rows // grid_split
        section_size_col = cols // grid_split
        sections = []
        for i in range(grid_split):
            for j in range(grid_split):
                section = heatmap_gray[i * section_size_row:(i + 1) * section_size_row,
                                         j * section_size_col:(j + 1) * section_size_col]
                sections.append(section)

        intensity_sums = [np.sum(section) for section in sections]
        guessed_fake_position = np.argmax(intensity_sums)
        total_positive_intensity = np.sum(heatmap_gray)
        intensity_weighted_accuracy = (intensity_sums[true_fake_pos] / total_positive_intensity
                                       if total_positive_intensity > 0 else 0.0)

        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy

class BCOSEvaluator:
    def __init__(self, model=None, device=None):
        """
        Initializes the evaluator with a model.
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        print(f"[DEBUG] Loaded {self.model.__class__.__name__} model onto {self.device}")

    def evaluate(self, tensor_list, path_list, grid_split):
        """
        Evaluates a list of grid tensors and returns the following lists:
        
            paths, original_images, heatmaps, model_outputs,
            guessed_fake_positions, intensity_sums, intensity_weighted_accuracies,
            true_fake_positions
        """
        results = []
        print(f"[DEBUG] Processing {len(tensor_list)} grids with grid_split={grid_split}.")

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            print(f"[DEBUG] Evaluating grid {idx} from file: {path}")

            # If tensor is 3-channel, convert it to 6-channel.
            if tensor.shape[1] == 3:
                tensor = torch.cat([tensor, 1.0 - tensor], dim=1)

            heatmap, output = self.generate_heatmap(tensor)
            
            # Extract true fake position BEFORE evaluating the heatmap.
            true_fake_pos = self.extract_fake_position(path)
            
            # Now pass it to evaluate_heatmap.
            guessed_fake_position, intensity_sums, accuracy = HeatmapEvaluator.evaluate_heatmap(
                heatmap, grid_split=grid_split, true_fake_pos=true_fake_pos
            )
            
            original_image = self.convert_to_numpy(tensor)



            result = {
                "path": path,
                "original_image": original_image,       # Original image in RGB, uint8
                "heatmap": heatmap,                       # RGBA heatmap
                "guessed_fake_position": guessed_fake_position,
                "accuracy": accuracy,
                "true_fake_position": true_fake_pos,
            }
            results.append(result)
            print(f"[DEBUG] Processed {os.path.basename(path)} with accuracy {accuracy:.4f}")

        return results

    def generate_heatmap(self, tensor):
        """
        Runs inference on a tensor, computes gradients, and generates a heatmap.
        """
        img = tensor.to(self.device).requires_grad_(True)
        print(f"[DEBUG] Input tensor shape: {img.shape}")

        self.model.zero_grad()
        out = self.model(img)
        print(f"[DEBUG] Model output: {out}")

        out.backward()
        grad = img.grad[0]
        print(f"[DEBUG] Gradients: min={grad.min().item()}, max={grad.max().item()}, mean={grad.mean().item()}")

        heatmap = HeatmapEvaluator.grad_to_img(img[0], grad, alpha_percentile=100)
        print(f"[DEBUG] Heatmap stats: shape={heatmap.shape}, min={heatmap.min()}, max={heatmap.max()}")
        return to_numpy(heatmap), out

    def convert_to_numpy(self, tensor):
        """
        Converts a tensor (with shape [1,6,H,W] or [6,H,W]) to an RGB image as a NumPy array.
        """
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # Now shape: [6, H, W]
        return np.array(to_numpy(tensor[:3].permute(1, 2, 0)) * 255, dtype=np.uint8)

    def extract_fake_position(self, path):
        """
        Extracts the fake position number from the filename.
        """
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            print(f"[DEBUG] Could not extract fake position from '{path}': {e}")
            return -1