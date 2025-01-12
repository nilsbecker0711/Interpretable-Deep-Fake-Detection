
import torch
import numpy as np
from b_cos.resnet import resnet50
from b_cos.bcosconv2d import BcosConv2d
from PIL import Image
import os


def preprocess_grid_to_heatmap(grid_path, target_channels=6):
    """
    Convert a 2x2 grid image into a 6-channel heatmap input for BCos ResNet50.

    Args:
        grid_path (str): Path to the grid image.
        target_channels (int): Number of output channels, default is 6.

    Returns:
        np.ndarray: Preprocessed heatmap.
    """
    with Image.open(grid_path) as img:
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize
        heatmap = np.repeat(img_array[..., None], target_channels, axis=-1)  # Create 6 channels
        return heatmap


def evaluate_bcos_resnet50(model, grid_paths, target_size=(128, 128)):
    """
    Evaluate BCos ResNet50 using grid images and the pointing game.

    Args:
        model (torch.nn.Module): Pre-trained BCos ResNet50 model.
        grid_paths (list): List of paths to 2x2 grid images.
        target_size (tuple): Size to resize each grid.

    Returns:
        dict: Evaluation results including accuracy and heatmap-based scores.
    """
    model.eval()
    all_scores = []

    for grid_path in grid_paths:
        heatmap = preprocess_grid_to_heatmap(grid_path)
        heatmap = torch.tensor(heatmap).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(heatmap)
            score = torch.nn.functional.softmax(output, dim=1).max().item()
            all_scores.append(score)

    avg_score = np.mean(all_scores)
    return {"average_score": avg_score, "total_images": len(grid_paths)}
