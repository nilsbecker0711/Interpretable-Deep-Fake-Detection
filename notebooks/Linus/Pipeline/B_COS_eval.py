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

from b_cos.resnet import resnet50






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







def check_and_resize(image, target_size):
    """
    Checks if the image's size (height, width) matches target_size. 
    If not, it resizes the image using PIL and returns the resized image.
    
    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        target_size (tuple): Desired size as (target_height, target_width).
    
    Returns:
        np.ndarray: The resized image array.
    """
    current_size = image.shape[:2]
    if current_size != target_size:
        print(f"Warning: image size {current_size} does not match target size {target_size}. Resizing.")
        # If image is normalized (values between 0 and 1), convert to uint8 first.
        if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        pil_img = Image.fromarray(image_uint8)
        # PIL resize expects (width, height)
        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
        resized = np.array(pil_img)
        # If the original image was normalized, convert it back.
        if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
            resized = resized.astype(np.float32) / 255.0
        return resized
    else:
        return image







# Helper function to convert a tensor to a NumPy array.
def to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()







# New complex heatmap evaluation classes.
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
    def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9, upscale_factor=3):
        """
        Evaluates a heatmap using intensity-weighted accuracy based on positive attributions.
        Instead of cropping, the function scales the heatmap up by upscale_factor and then splits it
        into grid_split x grid_split sections.

        Args:
            heatmap (numpy.ndarray): 3D numpy array (H, W, C) representing the heatmap.
            grid_split (int): Number of splits along each dimension (default: 3 for a 3x3 grid).
            top_percentile (float): Percentile for capping intensity outliers (default: 99.9).
            upscale_factor (int): Factor by which to scale up the heatmap before splitting (default: 3).

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

        # Upscale the heatmap using PIL.
        from PIL import Image
        original_size = heatmap_gray.shape  # (H, W)
        new_size = (int(original_size[1] * upscale_factor), int(original_size[0] * upscale_factor))  # (width, height)
        pil_img = Image.fromarray((heatmap_gray * 255).astype(np.uint8))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        heatmap_gray = np.array(pil_img).astype(np.float32) / 255.0

        # Now ensure the new dimensions are divisible by grid_split.
        rows, cols = heatmap_gray.shape
        if rows % grid_split != 0 or cols % grid_split != 0:
            # Adjust new dimensions to be divisible by grid_split (by cropping minimally).
            new_rows = grid_split * (rows // grid_split)
            new_cols = grid_split * (cols // grid_split)
            heatmap_gray = heatmap_gray[:new_rows, :new_cols]
            rows, cols = heatmap_gray.shape

        print(f"Adjusted heatmap size to {heatmap_gray.shape} to be divisible by {grid_split}.")

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
        intensity_weighted_accuracy = (intensity_sums[guessed_fake_position] / total_positive_intensity) if total_positive_intensity > 0 else 0.0

        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy







class DeepFakeEvaluator:




    def __init__(self, model_path, device=None):
        """
        Initialize the evaluator with a custom BCos ResNet50 model.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
    


    def visualize(self, img_np, heatmap, true_fake_pos, guessed_fake_position, intensity_sums, intensity_weighted_accuracy, grid_split=3):
        """
        Visualize the image and heatmap overlay with grid lines determined by grid_split.
        
        The original image is shown on the left (extent: (0, 224, 0, 224)).
        The heatmap is shown on the right (extent: (224, 448, 0, 224)).
        Grid lines are drawn over the heatmap region so that it is split into grid_split x grid_split cells.
        
        Args:
            img_np (numpy.ndarray): Original image (in RGB) as a NumPy array.
            heatmap (numpy.ndarray): The heatmap to overlay.
            true_fake_pos (int): The true fake image position extracted from the filename.
            guessed_fake_position (int): The section index with maximum intensity.
            intensity_sums (list): List of intensity sums for each grid section.
            intensity_weighted_accuracy (float): Computed intensity-weighted accuracy.
            grid_split (int): Number of splits along each dimension (e.g., 3 for a 3x3 grid).
        """
        fig, ax = plt.subplots(1, figsize=(8, 4))
        
        # Display the original image on the left.
        plt.imshow(img_np, extent=(0, 224, 0, 224))
        
        # Display the heatmap on the right.
        plt.imshow(heatmap, extent=(224, 448, 0, 224), alpha=0.6)
        
        # Determine the extent of the heatmap image.
        x_min, x_max = 224, 448
        y_min, y_max = 0, 224
        
        # Compute cell dimensions.
        cell_width = (x_max - x_min) / grid_split
        cell_height = (y_max - y_min) / grid_split
        
        # Draw horizontal grid lines (there will be grid_split-1 lines).
        for i in range(1, grid_split):
            y_pos = y_min + i * cell_height
            plt.hlines(y_pos, x_min, x_max, colors='grey', linestyles='dashed', linewidth=0.5)
        
        # Draw vertical grid lines (grid_split-1 lines).
        for j in range(1, grid_split):
            x_pos = x_min + j * cell_width
            plt.vlines(x_pos, y_min, y_max, colors='grey', linestyles='dashed', linewidth=0.5)
        
        title = f"True Fake Pos: {true_fake_pos}, Guessed: {guessed_fake_position}, Intensity Sums: {intensity_sums}"
        plt.title(title)
        plt.text((x_min + x_max) / 2, -10, f"Intensity-Weighted Accuracy: {intensity_weighted_accuracy:.4f}",
                fontsize=12, ha='center', va='top')
        
        # Optionally remove axes spines.
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.show()



    def evaluate(self, tensor_list, path_list, grid_split):
        """
        Evaluates a list of preprocessed tensors (and their corresponding file paths),
        generating heatmaps and visualizing the results.
        """
        print(f"Processing {len(tensor_list)} grids.")
        for tensor, path in zip(tensor_list, path_list):
            img = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            self.model.zero_grad()
            out = self.model(img)
            out.backward()

            # Heatmap generieren:
            heatmap = HeatmapEvaluator.grad_to_img(img[0], img.grad[0], alpha_percentile=100, smooth=5)
            heatmap[..., -1] *= to_numpy(out.sigmoid())
            heatmap = to_numpy(heatmap)

            # Hier setzt du den neuen Aufruf ein:
            guessed_fake_position, intensity_sums, intensity_weighted_accuracy = HeatmapEvaluator.evaluate_heatmap(heatmap, top_percentile=99.9, grid_split=grid_split)     

            # Konvertiere das Input-Bild für die Visualisierung:
            img_np = np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8)

            # Falls der Dateiname die Fake-Position enthält:
            try:
                true_fake_pos = int(os.path.basename(path).split('_fake_')[1].split('.')[0])
            except Exception:
                true_fake_pos = -1

            # Visualisiere die Ergebnisse:
            self.visualize(img_np, heatmap, true_fake_pos, guessed_fake_position, intensity_sums, intensity_weighted_accuracy)
            print(f"Accuracy for {path}: {intensity_weighted_accuracy:.4f}")