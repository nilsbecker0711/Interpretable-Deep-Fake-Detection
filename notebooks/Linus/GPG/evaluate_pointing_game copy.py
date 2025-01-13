import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Append your analysis path
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')
from b_cos.resnet import resnet50

# Utility functions
def to_numpy(tensor):
    """Converts a PyTorch tensor to a numpy array."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


# Transformation classes
class MyToTensor(transforms.ToTensor):
    """Custom transformation that ensures tensors are properly converted."""
    def __call__(self, input_img):
        if not isinstance(input_img, torch.Tensor):
            return super().__call__(input_img)
        return input_img


class AddInverse(nn.Module):
    """Adds the inverse of input channels to the tensor."""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        return torch.cat([in_tensor, 1 - in_tensor], self.dim)


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        """
        Args:
            folder_paths (dict): Dictionary of folder paths and corresponding labels.
            transform (callable, optional): Transform to be applied on the image.
        """
        self.image_files = []
        for fp, label in folder_paths.items():
            subfolders = [os.path.join(fp, d) for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
            self.image_files.extend(
                [(os.path.join(fp, f), label) for f in os.listdir(fp) if f.endswith((".png", ".jpg"))]
            )
            for folder_path in subfolders:
                self.image_files.extend(
                    [(os.path.join(folder_path, f), label) for f in os.listdir(folder_path) if f.endswith((".png", ".jpg"))]
                )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, label = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# Heatmap Processing and Evaluation
class HeatmapEvaluator:
    @staticmethod
    def grad_to_img(img, linear_mapping, smooth=15, alpha_percentile=99.5):
        """
        Compute a color image from dynamic linear mapping of B-cos models.
        """
        contribs = (img * linear_mapping).sum(0, keepdim=True)
        contribs = contribs[0]
        rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12)).clamp(0)
        rgb_grad = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))
        alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        if smooth:
            alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
        alpha = to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)
        rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
        return rgb_grad.transpose((1, 2, 0))

    @staticmethod
    def evaluate_heatmap(heatmap, white_pixel_threshold=255):
        """
        Splits a heatmap into 4 square sections, counts the white pixels in each,
        and calculates the accuracy score.

        Args:
            heatmap (numpy.ndarray): 2D numpy array representing the heatmap.
            white_pixel_threshold (int): Threshold to classify a pixel as white (default: 255).

        Returns:
            tuple: (least_white_index, white_pixel_counts, accuracy)
        """
        # Convert the heatmap to grayscale
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)  # Average R, G, B channels
        
        # Normalize to 255 if values are in the range [0, 1]
        if heatmap_gray.max() <= 1.0:
            heatmap_gray = (heatmap_gray * 255).astype(np.uint8)
        
        # Ensure the heatmap is square and divisible by 2
        rows, cols = heatmap_gray.shape
        if rows != cols or rows % 2 != 0:
            raise ValueError("The heatmap dimensions must be square and divisible by 2.")
        
        # Split the heatmap into 4 sections
        half = rows // 2
        sections = [
            heatmap_gray[:half, :half],  # Top-left (0)
            heatmap_gray[:half, half:],  # Top-right (1)
            heatmap_gray[half:, :half],  # Bottom-left (2)
            heatmap_gray[half:, half:]   # Bottom-right (3)
        ]
        
        # Count white pixels in each section
        white_pixel_counts = [np.sum(section >= white_pixel_threshold) for section in sections]

        # Total pixels in each section
        total_pixels_per_section = sections[0].size  # Same for all sections since they're equally sized

        # Find the section with the least white pixels (guessed fake position)
        least_white_index = np.argmin(white_pixel_counts)

        # Accuracy calculation
        fake_pixels = total_pixels_per_section  # All pixels in the fake section
        white_pixels_fake = white_pixel_counts[least_white_index]
        white_pixels_real = sum([white_pixel_counts[i] for i in range(4) if i != least_white_index])
        total_pixels = total_pixels_per_section * 4

        accuracy = (fake_pixels - white_pixels_fake + white_pixels_real) / total_pixels

        return least_white_index, white_pixel_counts, accuracy


# Model and Visualization
class DeepFakeEvaluator:
    def __init__(self, model_path, device=None):
        """
        Initialize the evaluator with a pre-trained model.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def evaluate(self, dataloader):
        """
        Evaluates a batch of images, generates heatmaps, and visualizes results.
        """
        for img_batch, label_batch, path_batch in dataloader:
            for i, (img, label, img_path) in enumerate(zip(img_batch, label_batch, path_batch)):
                img = img.unsqueeze(0).to(self.device).requires_grad_(True)  # Verwende self.device
                self.model.zero_grad()
                out = self.model(img)
                out.backward()

                # Generate the attention map
                att = HeatmapEvaluator.grad_to_img(img[0], img.grad[0], alpha_percentile=100, smooth=5)
                att[..., -1] *= to_numpy(out.sigmoid())
                att = to_numpy(att)

                # Convert the image to a NumPy array for visualization
                img_np = np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8)

                # Get the true fake position
                true_fake_pos = int(img_path.split('_fake_')[1].split('.')[0])

                # Evaluate the heatmap to find the guessed fake position and accuracy
                guess_pos, white_pixel_counts, accuracy = HeatmapEvaluator.evaluate_heatmap(att)

                # Visualize the results
                self.visualize(img_np, att, true_fake_pos, guess_pos, white_pixel_counts, accuracy)

                # Print accuracy
                print(f"Accuracy for {img_path}: {accuracy:.4f}")

    @staticmethod
    def visualize(img_np, att, true_fake_pos, guess_pos, white_pixel_counts, accuracy):
        """
        Visualizes the image and heatmap with additional header information.

        Args:
            img_np (numpy.ndarray): Original image as a NumPy array.
            att (numpy.ndarray): Attention map as a NumPy array.
            true_fake_pos (int): The actual fake image position in the grid.
            guess_pos (int): The guessed fake image position based on the heatmap.
            white_pixel_counts (list): White pixel counts for each grid section.
            accuracy (float): Accuracy score for the evaluation.
        """
        fig, ax = plt.subplots(1, figsize=(8, 4))

        # Show original image
        plt.imshow(img_np, extent=(0, 224, 0, 224))

        # Show heatmap
        plt.imshow(att, extent=(224, 2 * 224, 0, 224), alpha=0.6)

        # Add grid lines to separate heatmap sections
        plt.hlines(112, 224, 448, colors='grey', linestyles='dashed', linewidth=0.5)
        plt.vlines(336, 0, 224, colors='grey', linestyles='dashed', linewidth=0.5)

        # Adjust plot settings
        plt.xlim(0, 448)
        plt.xticks([])
        plt.yticks([])

        # Add title with true fake position, guessed position, and white pixel counts
        title = (
            f"True Fake Position: {true_fake_pos}, "
            f"Guessed Fake Position: {guess_pos}, "
            f"White Pixel Distribution: {white_pixel_counts}"
        )
        plt.title(title)

        # Add accuracy below the figure
        plt.text(
            224, -10, f"Accuracy: {accuracy:.4f}",
            fontsize=12, ha='center', va='top'
        )

        # Remove spines for a clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.show()


# Main Execution
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        MyToTensor(),
        AddInverse(dim=0),
    ])

    file_path_deepfakebench = {'/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/datasets/2x2_images': 1}
    dataset = CustomImageDataset(file_path_deepfakebench, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model_path = "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth"
    evaluator = DeepFakeEvaluator(model_path)
    evaluator.evaluate(dataloader)