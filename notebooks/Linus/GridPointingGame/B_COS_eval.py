import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Add the analysis path so that the b_cos modules are found.
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')
from b_cos.resnet import resnet50 as model


def preprocess_image(image_path):
    """
    Loads an image from the given file path, converts it to RGB, 
    and normalizes it to a float32 numpy array with values in [0, 1].

    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Image as a normalized (float32) numpy array.
    """
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img).astype(np.float32) / 255.0
    return np_img

def to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()

class HeatmapEvaluator:
    @staticmethod
    def grad_to_img(img, linear_mapping, smooth=15, alpha_percentile=99.5):
        """
        Compute a color image from a dynamic linear mapping of BCos models.
        """
        contribs = (img * linear_mapping).sum(0, keepdim=True)[0]
        rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12)).clamp(0)
        rgb_grad = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))
        alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        if smooth:
            alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
        alpha = to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
        rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
        return rgb_grad.transpose((1, 2, 0))

    @staticmethod
    def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9):
        """
        Evaluates a heatmap using intensity-weighted accuracy based on positive attributions.
        Instead of cropping, the function scales the heatmap up and then splits it
        into grid_split x grid_split sections.

        Args:
            heatmap (numpy.ndarray): 3D numpy array (H, W, C) representing the heatmap.
            grid_split (int): Number of splits along each dimension (default: 3 for a 3x3 grid).
            top_percentile (float): Percentile for capping intensity outliers (default: 99.9).

        Returns:
            tuple: (guessed_fake_position, intensity_sums, intensity_weighted_accuracy)
        """
        # Convert heatmap to grayscale by averaging the first 3 channels.
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)
        
        # Normalize intensities to [0,1].
        if heatmap_gray.max() > 1.0:
            heatmap_gray = heatmap_gray / 255.0

        intensity_cap = np.percentile(heatmap_gray, top_percentile)
        heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

        # Upscale the heatmap using PIL if necessary.
        original_size = heatmap_gray.shape  # (H, W)
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)  # (width, height)
            print(f"[DEBUG] Upscaling heatmap from {original_size} to {new_size} for proper grid division.")
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
        intensity_weighted_accuracy = (
            intensity_sums[guessed_fake_position] / total_positive_intensity
            if total_positive_intensity > 0 else 0.0
        )

        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy

class BCOSEvaluator:
    def __init__(self, model_path, device=None):
        """Initialize the evaluator with a custom BCos ResNet50 model."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[DEBUG] Loaded BCos model from '{model_path}' onto device '{self.device}'.")

    def evaluate(self, tensor_list, path_list, grid_split):
        """Evaluates a list of preprocessed tensors and generates heatmaps."""
        print(f"[DEBUG] Processing {len(tensor_list)} grids with grid_split={grid_split}.")

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            print(f"[DEBUG] Evaluating grid {idx} from file: {path}")
            heatmap, output = self.generate_heatmap(tensor)
            guessed_fake_position, intensity_sums, accuracy = HeatmapEvaluator.evaluate_heatmap(
                heatmap, grid_split=grid_split
            )

            img_np = self.convert_to_numpy(tensor)
            true_fake_pos = self.extract_fake_position(path)

            self.visualize(
                img_np, heatmap, true_fake_pos, guessed_fake_position,
                intensity_sums, accuracy, grid_split
            )
            print(f"[DEBUG] Accuracy for '{path}': {accuracy:.4f}")

    def generate_heatmap(self, tensor):
        """Runs inference, computes gradients, and generates a heatmap."""
        img = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        print(f"[DEBUG] Model input shape: {img.shape}")

        self.model.zero_grad()
        out = self.model(img)
        out.backward()

        heatmap = HeatmapEvaluator.grad_to_img(img[0], img.grad[0], alpha_percentile=100, smooth=5)
        heatmap[..., -1] *= to_numpy(out.sigmoid())
        return to_numpy(heatmap), out

    def convert_to_numpy(self, tensor):
        """Converts a PyTorch tensor image to a NumPy array."""
        return np.array(to_numpy(tensor[:3].permute(1, 2, 0)) * 255, dtype=np.uint8)

    def extract_fake_position(self, path):
        """Extracts the true fake position from the filename."""
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            print(f"[DEBUG] Could not extract fake position from '{path}': {e}")
            return -1

    def visualize(self, img_np, heatmap, true_fake_pos, guessed_fake_position, intensity_sums, intensity_weighted_accuracy, grid_split):
        """Visualizes the heatmap overlay with grid lines using the correct grid_split value."""
        print(f"[DEBUG] Visualizing heatmap with shape: {heatmap.shape}")
        fig, ax = plt.subplots(1, figsize=(8, 4))
        
        # Display the original image on the left.
        plt.imshow(img_np, extent=(0, 224, 0, 224))
        
        # Display the heatmap on the right.
        plt.imshow(heatmap, extent=(224, 448, 0, 224), alpha=1)
        
        # Define heatmap display bounds.
        x_min, x_max = 224, 448
        y_min, y_max = 0, 224
        
        # Draw grid lines.
        for i in range(1, grid_split):
            y_pos = y_min + (i / grid_split) * (y_max - y_min)
            plt.hlines(y_pos, x_min, x_max, colors='grey', linestyles='dashed', linewidth=0.5)
        for j in range(1, grid_split):
            x_pos = x_min + (j / grid_split) * (x_max - x_min)
            plt.vlines(x_pos, y_min, y_max, colors='grey', linestyles='dashed', linewidth=0.5)

        # Add text for title and accuracy.
        title_x_pos = x_min - 220  # Shift further left
        title_y_pos = y_max + 20   # Positioned above the image

        plt.text(title_x_pos, title_y_pos,
                 f"True Fake Pos: {true_fake_pos}, Guessed: {guessed_fake_position}",
                 fontsize=12, ha='left', va='center', fontweight="bold")

        plt.text(title_x_pos, title_y_pos - 10,
                 f"Intensity-Weighted Accuracy: {intensity_weighted_accuracy:.4f}",
                 fontsize=10, ha='left', va='center')

        plt.text(title_x_pos, title_y_pos - 260,
                 f"Intensity Sums: {intensity_sums}",
                 fontsize=10, ha='left', va='center')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.show()