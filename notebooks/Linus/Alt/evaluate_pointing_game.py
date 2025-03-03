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

# Analysis path
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')
from b_cos.resnet import resnet50

# Utility
def to_numpy(tensor):
    """Converts a PyTorch tensor to a numpy array."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


# Transformation
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


# Custom Dataset
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


# Heatmap processing and evaluation
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
    def evaluate_heatmap(heatmap, top_percentile=99.9):
        """
        Evaluates a heatmap using intensity-weighted accuracy based on positive attributions.

        Args:
            heatmap (numpy.ndarray): 3D numpy array (H, W, C) representing the heatmap.
            top_percentile (float): Percentile for capping intensity outliers (default: 99.9).

        Returns:
            tuple: (guessed_fake_position, intensity_sums, intensity_weighted_accuracy)
        """
        # Convert heatmap to grayscale by averaging R, G, B channels
        heatmap_gray = np.mean(heatmap[..., :3], axis=-1)

        # Normalize heatmap intensities to [0, 1]
        if heatmap_gray.max() > 1.0:
            heatmap_gray = heatmap_gray / 255.0

        # Cap intensities at the top_percentile to handle outliers
        intensity_cap = np.percentile(heatmap_gray, top_percentile)
        heatmap_gray = np.clip(heatmap_gray, 0, intensity_cap) / intensity_cap

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

        # Calculate the sum of positive intensities in each section
        intensity_sums = [np.sum(section) for section in sections]

        # Find the section with the maximum positive attribution (guessed fake position)
        guessed_fake_position = np.argmax(intensity_sums)

        # Calculate the total positive intensity across the entire heatmap
        total_positive_intensity = np.sum(heatmap_gray)

        # Calculate intensity-weighted accuracy
        if total_positive_intensity > 0:
            intensity_weighted_accuracy = intensity_sums[guessed_fake_position] / total_positive_intensity
        else:
            intensity_weighted_accuracy = 0.0

        return guessed_fake_position, intensity_sums, intensity_weighted_accuracy

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
                # Ensure device is properly referenced
                img = img.unsqueeze(0).to(self.device).requires_grad_(True)
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

                # Evaluate the heatmap to find guessed fake position and intensity-weighted accuracy
                guess_pos, intensity_sums, accuracy = HeatmapEvaluator.evaluate_heatmap(att)

                # Visualize the results
                self.visualize(img_np, att, true_fake_pos, guess_pos, intensity_sums, accuracy)

                # Print accuracy
                print(f"Accuracy for {img_path}: {accuracy:.4f}")

    @staticmethod
    def visualize(img_np, att, true_fake_pos, guess_pos, intensity_sums, accuracy):
        """
        Visualizes the image and heatmap with additional header information.

        Args:
            img_np (numpy.ndarray): Original image as a NumPy array.
            att (numpy.ndarray): Attention map as a NumPy array.
            true_fake_pos (int): The actual fake image position in the grid.
            guess_pos (int): The guessed fake image position based on the heatmap.
            intensity_sums (list): List of positive intensity sums for each grid section.
            accuracy (float): Intensity-weighted accuracy score for the heatmap.
        """
        fig, ax = plt.subplots(1, figsize=(8, 4))

        # Show original image
        plt.imshow(img_np, extent=(0, 224, 0, 224))

        # Show heatmap
        plt.imshow(att, extent=(224, 2 * 224, 0, 224), alpha=0.6)

        # Add grid lines
        plt.hlines(112, 224, 448, colors='grey', linestyles='dashed', linewidth=0.5)
        plt.vlines(336, 0, 224, colors='grey', linestyles='dashed', linewidth=0.5)

        # Add title and accuracy
        title = (
            f"True Fake Position: {true_fake_pos}, "
            f"Guessed Fake Position: {guess_pos}, "
            f"Intensity Sums: {intensity_sums}"
        )
        plt.title(title)
        plt.text(224, -10, f"Intensity-Weighted Accuracy: {accuracy:.4f}", fontsize=12, ha='center', va='top')

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.show()




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