from B_COS_eval import HeatmapEvaluator, DeepFakeEvaluator, preprocess_image, check_and_resize
import sys
import torch
import os
import numpy as np
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add the analysis path so that the b_cos modules are found.
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')# Import the custom ResNet50 used by the BCos model.
from b_cos.resnet import resnet50

def main(xai_method, base_output_dir, model_path, target_height, target_width):
    # Determine the target channel format based on the chosen XAI method.
    if xai_method.lower() == "bcos":
        target_channels = 6
    elif xai_method.lower() in ["lime", "gradcam"]:
        target_channels = 3
    else:
        raise ValueError("Unknown xai_method specified. Choose 'bcos', 'lime', or 'gradcam'.")
    
    # Just a check if the size is right
    expected_size = (target_height, target_width)
    
    # Debugg
    print(f"Using XAI method '{xai_method}' -> target_channels set to {target_channels}")
    print(f"Expected image size: {expected_size}")
    
    # Choose the subfolder based on target_channels and list grid file paths.
    if target_channels == 3:
        grid_dir = os.path.join(base_output_dir, "3ch")
        # For 3-channel grids, we expect PNG files.
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.png')]
    elif target_channels == 6:
        grid_dir = os.path.join(base_output_dir, "6ch")
        # For 6-channel grids, we expect NumPy (.npy) files.
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.npy')]
    else:
        raise ValueError("target_channels must be either 3 or 6.")
    
    print(f"Found {len(grid_paths)} grids for evaluation in {grid_dir}.")
    
    preprocessed_grids = []
    for grid_path in grid_paths:
        if grid_path.endswith('.npy'):
            # For 6-channel grids: load the NumPy file.
            grid = np.load(grid_path)
            # Assume that the saved 6-channel grid is already correctly sized
            # and that the inverse was computed during creation.
            grid = grid.astype(np.float32)
        else:
            # For 3-channel grids: load the image using preprocess_image.
            grid = preprocess_image(grid_path)
        preprocessed_grids.append(grid)
    
    # Convert each preprocessed grid (assumed to be in shape (H, W, C)) into a PyTorch tensor with shape (C, H, W).
    preprocessed_tensors = []
    for grid in preprocessed_grids:
        if grid.ndim == 3:
            grid = np.transpose(grid, (2, 0, 1))
        tensor = torch.tensor(grid, dtype=torch.float32)
        preprocessed_tensors.append(tensor)

   # Instead of calling evaluate_bcos_resnet50, we now use DeepFakeEvaluator.
    evaluator = DeepFakeEvaluator(model_path)
    # Since you're not using a DataLoader, iterate over your lists directly.
    evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=args.grid_split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    parser.add_argument(
        "--xai_method", 
        type=str, 
        default="bcos",
        choices=["bcos", "lime", "gradcam"],
        help="XAI method to use: 'bcos' uses 6ch; 'lime' uses 3ch; 'gradcam' defaults to 3ch."
    )
    parser.add_argument(
        "--base_output_dir", 
        type=str, 
        default="datasets/2x2_grids",
        help="Base directory where grids are saved."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth",
        help="Path to the model file."
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=224,
        help="Target height for evaluation images."
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=224,
        help="Target width for evaluation images."
    )

    ### doesnt work####
    parser.add_argument(
        "--grid_split",
        type=int,
        default=3,
        help="Number of splits along each dimension for evaluating the heatmap (e.g., 3 for a 3x3 grid)."
    )
    
    args = parser.parse_args()
    main(
        xai_method=args.xai_method, 
        base_output_dir=args.base_output_dir, 
        model_path=args.model_path,
        target_height=args.target_height,
        target_width=args.target_width
    )