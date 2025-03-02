import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from B_COS_eval import BCOSEvaluator  # Assuming you’re evaluating using BCos.
# from LIME_eval import LimeEvaluator
# from GRADCAM_eval import GradCamEvaluator
## still need 500 best and so on
## lime and gradcam
## mappointinggame
## dataloader json fix train also der dataloader greift dadrauf zu
# 20 most accurat per model to create grids

def parse_arguments():
    """Parse command-line arguments to override default settings."""
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    
    defaults = {
        "xai_method": "bcos",
        "base_output_dir": "datasets/GPG_grids",
        "model_path": "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth",
        "grid_split": 3,
    }
    
    parser.add_argument("--xai_method", type=str, default=defaults["xai_method"], choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--base_output_dir", type=str, default=defaults["base_output_dir"], help="Base directory where grids are saved.")
    parser.add_argument("--model_path", type=str, default=defaults["model_path"], help="Path to the model file.")
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

    # Select grid directory based on XAI method.
    # (Assuming all grids are now saved as tensor files with .pt extension.)
    if xai_method == "bcos":
        grid_dir = os.path.join(base_output_dir, "6ch")
    else:
        grid_dir = os.path.join(base_output_dir, "3ch")
    grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]

    print(f"Found {len(grid_paths)} grids for evaluation in {grid_dir}.")

    # Directly load the tensors.
    preprocessed_tensors = []
    for grid_path in grid_paths:
        grid_tensor = torch.load(grid_path)
        print(f"[DEBUG] Loaded grid tensor from {grid_path} with shape: {grid_tensor.shape}")
        preprocessed_tensors.append(grid_tensor)

    # Run evaluation using the selected XAI method.
    if xai_method == "bcos":
        evaluator = BCOSEvaluator(model_path)
        evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)
    # elif xai_method == "lime":
    #     lime_evaluator = LimeEvaluator(model_path, model_name)  # Adapt as needed.
    #     lime_evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)
    # elif xai_method == "gradcam":
    #     gradcam_evaluator = GradCamEvaluator(model_path, model_name)  # Adapt as needed.
    #     gradcam_evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=grid_split)

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py