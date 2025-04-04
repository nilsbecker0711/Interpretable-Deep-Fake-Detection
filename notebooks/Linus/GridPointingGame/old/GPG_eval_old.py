import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from B_COS_eval import BCOSEvaluator, preprocess_image
# from LIME_eval import LimeEvaluator
# from GRADCAM_eval import GradCamEvaluator
## still need 500 best and so on
## lime and gradcam
## mappointinggame
## dataloader json fix train also der dataloader greift dadrauf zu



# Append the analysis path so that the b_cos modules are found.
sys.path.append('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/analysis')
from b_cos.resnet import resnet50 as model  # Change the model depending on the one you want to use !!!!!!!!!!!!


def parse_arguments():
    """Parse command-line arguments to override default settings."""
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    
    defaults = {
        "xai_method": "bcos",
        "base_output_dir": "datasets/GPG_grids",
        "model_path": "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth",
        # "target_height": 224,
        # "target_width": 224,
        "grid_split": 3,
    }
    
    parser.add_argument("--xai_method", type=str, default=defaults["xai_method"], choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--base_output_dir", type=str, default=defaults["base_output_dir"], help="Base directory where grids are saved.")
    parser.add_argument("--model_path", type=str, default=defaults["model_path"], help="Path to the model file.")
    # parser.add_argument("--target_height", type=int, default=defaults["target_height"], help="Target height for evaluation images.")
    # parser.add_argument("--target_width", type=int, default=defaults["target_width"], help="Target width for evaluation images.")
    parser.add_argument("--grid_split", type=int, default=defaults["grid_split"], help="Grid size for heatmap evaluation.")
    
    return parser.parse_args()


def main():
    """Main function to execute the evaluation pipeline."""
    args = parse_arguments()

    # Assign variables from arguments.
    xai_method = args.xai_method
    base_output_dir = args.base_output_dir
    model_path = args.model_path
    grid_split = args.grid_split

    print(f"Using XAI method: {xai_method}")
    print(f"Evaluating from directory: {base_output_dir}")
    print(f"Model path: {model_path}")
    print(f"Grid split: {grid_split}x{grid_split}")

    # Load grids based on XAI method.
    if xai_method == "bcos":
        grid_dir = os.path.join(base_output_dir, "6ch")
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.npy')]
    else:
        grid_dir = os.path.join(base_output_dir, "3ch")
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.png')]

    print(f"Found {len(grid_paths)} grids for evaluation in {grid_dir}.")

    # Preprocess the grids.
    preprocessed_grids = []
    for grid_path in grid_paths:
        if grid_path.endswith('.npy'):
            grid = np.load(grid_path).astype(np.float32)
        else:
            grid = preprocess_image(grid_path)
        print(f"Preprocessed grid shape before tensor conversion: {grid.shape}")  # Debugging Step
        preprocessed_grids.append(grid)

    # Convert grids to tensors.
    preprocessed_tensors = []
    for grid in preprocessed_grids:
        print(f"Shape before torch conversion: {grid.shape}")  # Debugging Step
        tensor = torch.tensor(np.transpose(grid, (2, 0, 1)), dtype=torch.float32)
        print(f"Tensor shape after conversion: {tensor.shape}")  # Debugging Step
        preprocessed_tensors.append(tensor)

    # Run evaluation using the selected XAI method.
    if xai_method == "bcos":
        bcos_evaluator = BCOSEvaluator(model_path)
        bcos_evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)
    elif xai_method == "lime":
        # Ensure you define 'model_name' or adapt the LimeEvaluator accordingly.
        lime_evaluator = LimeEvaluator(model_path, model_name)  # Pass model_name if needed.
        lime_evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)
    elif xai_method == "gradcam":
        # Ensure you define 'model_name' or adapt the GradCamEvaluator accordingly.
        gradcam_evaluator = GradCamEvaluator(model_path, model_name)  # Pass model_name if needed.
        gradcam_evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)


if __name__ == "__main__":
    main()


# python notebooks/Linus/GridPointingGame/GPG_eval.py