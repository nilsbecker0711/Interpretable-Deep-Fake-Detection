import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Append the analysis path so that the b_cos modules are found.
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')
from b_cos.resnet import resnet50 as model


def to_numpy(t):
    """
    Converts a tensor to a numpy array if it is not already one.
    """
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()

class HeatmapEvaluator:
    @staticmethod
    def grad_to_img(img, linear_mapping, alpha_percentile=99.5):  # smooth=15
        """
        Compute an RGBA heatmap from the dynamic linear mapping of BCos models.
        
        Steps:
        1. Compute overall contribution per pixel.
        2. Normalize the 6-channel linear mapping to [0,1].
        3. Split the 6 channels into two groups:
           - First three channels: positive contributions (r, g, b)
           - Last three channels: negative contributions (1 - r, 1 - g, 1 - b)
        4. Normalize each color by dividing the positive part by the sum of positive and negative parts.
        5. Compute an alpha channel based on the squared L2 norm of the mapping.
        6. Concatenate the normalized RGB with the alpha channel to form a final RGBA image.
        
        Args:
            img (torch.Tensor): Input image tensor (used for contribution calculation).
            linear_mapping (torch.Tensor): The dynamic mapping (6 channels) computed via gradients.
            alpha_percentile (float): Percentile used for normalizing the alpha channel.
        
        Returns:
            heatmap (np.ndarray): RGBA heatmap with shape [H, W, 4].
        """
        # (1) Compute overall contribution for each pixel
        contribs = (img * linear_mapping).sum(0, keepdim=True)[0]
        print(f"[DEBUG] Contribution stats: min={contribs.min().item()}, max={contribs.max().item()}")

        # (2) Normalize the linear mapping so values fall between 0 and 1.
        rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12)).clamp(0)
        print(f"[DEBUG] RGB grad (first 3 channels) stats: min={rgb_grad[:3].min().item()}, max={rgb_grad[:3].max().item()}")

        # (3) Normalize the first three channels by the sum of positive and negative parts.
        # This converts the 6-channel mapping to an RGB mapping with values in [0,1].
        rgb_grad_np = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))
        
        # (4) Compute the opacity (alpha) channel from the mapping.
        # Using the squared L2 norm emphasizes strong activations.
        alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
        # Zero out alpha for pixels with a negative overall contribution.
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        alpha = to_numpy(alpha)
        # Normalize the alpha channel by a high percentile to limit extreme values.
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
        print(f"[DEBUG] Alpha channel stats: min={alpha.min()}, max={alpha.max()}")

        # (5) Concatenate the normalized RGB mapping with the alpha channel.
        rgb_grad_np = np.concatenate([rgb_grad_np, alpha], axis=0)
        # Transpose the result from [6, H, W] to [H, W, 6] then select 4 channels to form an RGBA image.
        # In our case, the 6 channels now represent [R, G, B, alpha1] where we only need the first 3 and alpha.
        heatmap = rgb_grad_np.transpose((1, 2, 0))
        print(f"[DEBUG] Final heatmap shape: {heatmap.shape}")
        return heatmap


    @staticmethod
    def evaluate_heatmap(heatmap, grid_split=3, top_percentile=99.9):
        """
        Evaluates the heatmap using intensity-weighted accuracy based on positive attributions.
        The heatmap is converted to grayscale, normalized, upscaled if necessary,
        and then split into grid sections.
        
        Args:
            heatmap (np.ndarray): The RGBA heatmap (H, W, 4).
            grid_split (int): Number of grid splits along one dimension.
            top_percentile (float): Percentile for capping intensity values.
        
        Returns:
            guessed_fake_position (int): Grid cell index with the highest summed intensity.
            intensity_sums (list): Sum of intensities for each grid cell.
            intensity_weighted_accuracy (float): Accuracy computed as ratio of the max cell's intensity to total intensity.
        """
        # Convert to grayscale by averaging the RGB channels.
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)
        if heatmap_gray.max() > 1.0:
            heatmap_gray = heatmap_gray / 255.0

        # Cap extreme values and normalize.
        intensity_cap = np.percentile(heatmap_gray, top_percentile)
        heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

        # Upscale if needed to ensure even grid divisions.
        original_size = heatmap_gray.shape  # (H, W)
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
        else:
            new_size = original_size

        pil_img = Image.fromarray((heatmap_gray * 255).astype(np.uint8))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        heatmap_gray = np.array(pil_img).astype(np.float32) / 255.0

        # Divide the image into grid cells and sum the intensities.
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
        intensity_weighted_accuracy = (intensity_sums[guessed_fake_position] / total_positive_intensity
                                       if total_positive_intensity > 0 else 0.0)

        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy

class BCOSEvaluator:
    def __init__(self, model_path, device=None, xai_method="bcos"):
        """
        Initializes the evaluator with a model.
        
        Args:
            model_path (str): Path to the model weights.
            device (torch.device, optional): Device to run the model on.
            xai_method (str): Name of the XAI method (e.g., "bcos").
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[DEBUG] Loaded BCos model from '{model_path}' onto device '{self.device}'.")
        self.xai_method = xai_method

    def evaluate(self, tensor_list, path_list, grid_split):
        """
        Evaluates a list of grid tensors and saves visualizations and summary statistics.
        
        Args:
            tensor_list (list): List of preprocessed grid tensors.
            path_list (list): Corresponding file paths for each tensor.
            grid_split (int): Number of divisions for grid evaluation.
        """
        print(f"[DEBUG] Processing {len(tensor_list)} grids with grid_split={grid_split}.")

        # Create an output directory named based on the model and xai method.
        model_name = self.model.__class__.__name__
        output_dir = os.path.join("results", f"{model_name}_{self.xai_method}")
        os.makedirs(output_dir, exist_ok=True)

        # Lists to accumulate individual accuracy and guess values.
        acc_list = []
        guess_list = []

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            print(f"[DEBUG] Evaluating grid {idx} from file: {path}")
            heatmap, output = self.generate_heatmap(tensor)
            guessed_fake_position, intensity_sums, accuracy = HeatmapEvaluator.evaluate_heatmap(heatmap, grid_split=grid_split)
            acc_list.append(accuracy)
            guess_list.append(guessed_fake_position)

            img_np = self.convert_to_numpy(tensor)
            true_fake_pos = self.extract_fake_position(path)

            # Visualize the grid result and save it in the output directory.
            self.visualize(img_np, heatmap, true_fake_pos, guessed_fake_position,
                           intensity_sums, accuracy, grid_split, output_dir, idx)
            print(f"[DEBUG] Accuracy for '{path}': {accuracy:.4f}")

        # Compute overall averages and save a summary file.
        avg_acc = np.mean(acc_list)
        avg_guess = np.mean(guess_list)
        summary_text = (
            f"Total grids: {len(tensor_list)}\n"
            f"Average Intensity-Weighted Accuracy: {avg_acc:.4f}\n"
            f"Average Guessed Fake Position: {avg_guess:.2f}\n"
            f"Individual Accuracies: {acc_list}\n"
            f"Individual Guesses: {guess_list}\n"
        )
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"[DEBUG] Saved summary statistics to {summary_path}")

    def generate_heatmap(self, tensor):
        """
        Runs inference on a tensor, computes gradients, and generates a heatmap.
        
        Args:
            tensor (torch.Tensor): A preprocessed grid tensor.
        
        Returns:
            heatmap (np.ndarray): RGBA heatmap.
            out (torch.Tensor): Model output.
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
        print(f"[DEBUG] Heatmap stats before alpha modulation: shape={heatmap.shape}, min={heatmap.min()}, max={heatmap.max()}")
        # Do not modulate alpha by model output.
        print(f"[DEBUG] Heatmap stats after alpha modulation: min={heatmap.min()}, max={heatmap.max()}")

        print(f"[DEBUG] Heatmap RGB channel stats:")
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = heatmap[..., i]
            print(f"  {channel}: min={channel_data.min()}, max={channel_data.max()}, mean={channel_data.mean()}")

        return to_numpy(heatmap), out

    def convert_to_numpy(self, tensor):
        """
        Converts a tensor (with shape [1,6,H,W] or [6,H,W]) to an RGB image as a NumPy array.
        Takes the first 3 channels, scales to [0,255], and permutes to [H,W,3].
        """
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # Now shape: [6, H, W]
        return np.array(to_numpy(tensor[:3].permute(1, 2, 0)) * 255, dtype=np.uint8)

    def extract_fake_position(self, path):
        """
        Extracts the fake position number from the filename.
        Expects the filename to contain '_fake_' followed by a number.
        """
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            print(f"[DEBUG] Could not extract fake position from '{path}': {e}")
            return -1

    def visualize(self, img_np, heatmap, true_fake_pos, guessed_fake_position, intensity_sums,
                  intensity_weighted_accuracy, grid_split, output_dir, idx):
        """
        Visualizes the explanation side by side with the original image and gridlines,
        then saves the figure to the output directory.
        
        Args:
            img_np (np.ndarray): Original image (RGB, uint8 [0,255]).
            heatmap (np.ndarray): RGBA heatmap.
            true_fake_pos (int): True fake position (from filename).
            guessed_fake_position (int): Guessed fake position from evaluation.
            intensity_sums (list): List of intensity sums for each grid cell.
            intensity_weighted_accuracy (float): Calculated accuracy metric.
            grid_split (int): Number of grid divisions.
            output_dir (str): Directory to save visualizations.
            idx (int): Index of the current grid.
        """
        print(f"[DEBUG] Visualizing heatmap with shape: {heatmap.shape}")
        # Create a figure with 2 subplots: one for the original image and one for the explanation.
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Convert original image to [0,1] float for display.
        orig_img = img_np.astype(np.float32) / 255.0
        axs[0].imshow(orig_img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        # Display the explanation (heatmap) in the second subplot.
        axs[1].imshow(heatmap)
        axs[1].set_title("B-cos Explanation")
        axs[1].axis("off")

        # Draw gridlines on the explanation subplot.
        height, width, _ = heatmap.shape
        for i in range(1, grid_split):
            y = i * height / grid_split
            x = i * width / grid_split
            axs[1].hlines(y, 0, width, colors='grey', linestyles='dashed', linewidth=0.5)
            axs[1].vlines(x, 0, height, colors='grey', linestyles='dashed', linewidth=0.5)

        # Add a super-title with annotations.
        plt.suptitle(f"True Fake Pos: {true_fake_pos}, Guessed: {guessed_fake_position}\n"
                     f"Intensity-Weighted Accuracy: {intensity_weighted_accuracy:.4f}",
                     fontsize=12, fontweight="bold", color='black')

        # Save the figure instead of displaying it.
        save_path = os.path.join(output_dir, f"grid_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[DEBUG] Saved visualization to {save_path}")