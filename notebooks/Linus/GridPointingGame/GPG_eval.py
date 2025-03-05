#!/usr/bin/env python
import os
import sys
import torch
import argparse
import numpy as np
import yaml

# Import grid creation functions.
from GPG_creation import load_config, load_model, RankedGPGCreator

# Evaluators.
from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  # Adjust the import path if needed.
# from GRADCAM_eval import GradCamEvaluator  # Uncomment if available.

PROJECT_PATH = "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection"
sys.path.append(PROJECT_PATH)

XAI_METHOD = "lime"
BASE_OUTPUT_DIR = "datasets/GPG_grids"
GRID_SPLIT = 3
REAL_DIR = "datasets/FaceForensics++/original_sequences/actors/c40/frames"
FAKE_DIR = "datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"
MAX_GRIDS = 2
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception.yaml")

ADDITIONAL_ARGS = {
    "model_name": "xception",
    "test_batchSize": 12,
    "pretrained": os.path.join(PROJECT_PATH, "weights/resnet/ckpt_best.pth")
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    parser.add_argument("--real_dir", type=str, default=REAL_DIR, help="Directory with real images.")
    parser.add_argument("--fake_dir", type=str, default=FAKE_DIR, help="Directory with fake images.")
    parser.add_argument("--base_output_dir", type=str, default=BASE_OUTPUT_DIR, help="Base output directory for grids.")
    parser.add_argument("--max_grids", type=int, default=MAX_GRIDS, help="Maximum number of grids to create.")
    parser.add_argument("--xai_method", type=str, default=XAI_METHOD, choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model configuration file.")
    parser.add_argument("--grid_split", type=int, default=GRID_SPLIT, help="Grid size for evaluation (e.g. 3 for a 3x3 grid).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    grid_size = (args.grid_split, args.grid_split)
    print(f"XAI: {args.xai_method}, Base: {args.base_output_dir}, Model: {args.model_path}, Grid: {args.grid_split}x{args.grid_split}")
    print(f"Real: {args.real_dir}, Fake: {args.fake_dir}, Max grids: {args.max_grids}")
    
    # Load configuration.
    config = load_config(args.model_path, additional_args=ADDITIONAL_ARGS)
    
    # Load the model once.
    model = load_model(config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[DEBUG] Loaded {model.__class__.__name__} model onto {device}")
    
    # Extract model and weight names for output folder naming.
    model_name = config.get("model_name", "defaultModel")
    pretrained_path = config.get("pretrained", "default_weights.pth")
    weights_name = os.path.basename(pretrained_path).split('.')[0]
    
    # Instantiate the RankedGPGCreator with the loaded model.
    grid_creator = RankedGPGCreator(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        base_output_dir=args.base_output_dir,
        grid_size=grid_size,
        max_grids=args.max_grids,
        model=model,
        model_name=model_name,
        weights_name=weights_name
    )
    grid_creator.create_GPG_grids()

    # Optionally, you can inspect the fake images ranking:
    # print("Top-ranked fake images:", grid_creator.fake_images_ranked[:5])
    
    # Determine grid directory based on XAI method.

    grid_dir = os.path.join(grid_creator.output_folder, "6ch") if args.xai_method == "bcos" else os.path.join(grid_creator.output_folder, "3ch")
    grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]
    print(f"Found {len(grid_paths)} grid tensors for evaluation in {grid_dir}.")

    # Load grid tensors.
    preprocessed_tensors = []
    for grid_path in grid_paths:
        grid_tensor = torch.load(grid_path)
        print(f"[DEBUG] Loaded grid tensor from {grid_path} with shape: {grid_tensor.shape}")
        preprocessed_tensors.append(grid_tensor)

    # Instantiate the evaluator using the already loaded model.
    if args.xai_method == "bcos":
        evaluator = BCOSEvaluator(config, model=model)  # Adjust as needed.
    elif args.xai_method == "lime":
        evaluator = LIMEEvaluator(config, model=model)
    elif args.xai_method == "gradcam":
        # evaluator = GradCamEvaluator(config, model=model)  # Uncomment if implemented.
        print("GradCAM evaluator not implemented in this example.")
        return
    else:
        raise ValueError(f"Unknown xai_method: {args.xai_method}")

    # Run the evaluation.
    evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=args.grid_split)

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py