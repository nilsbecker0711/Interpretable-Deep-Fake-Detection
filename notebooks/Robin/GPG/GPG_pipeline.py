
from grid_creation import create_2x2_grids
from evaluation import evaluate_bcos_resnet50
from data_handling import get_all_png_files
from visualization import plot_heatmap

import torch
import os

def main():
    # Step 1: Define Paths
    real_dir = "datasets/FaceForensics++/original_sequences/actors/c40/frames"
    fake_dir = "datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"
    output_dir = "datasets/2x2_images"  # Directory to save grids
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Create Grids
    create_2x2_grids(real_dir=real_dir, fake_dir=fake_dir, output_dir=output_dir)
    print(f"2x2 grids created and saved in {output_dir}.")

    # Step 3: Evaluate Model
    # Load BCos ResNet50 model (replace with actual model path)
    model = torch.load("path/to/your/bcos_resnet50.pth")
    
    # Collect generated grids
    grid_paths = get_all_png_files(output_dir)
    
    # Evaluate model on the grids
    metrics = evaluate_bcos_resnet50(model, grid_paths)
    print(f"Evaluation Metrics: {metrics}")

    # Step 4: Visualize Results (Optional)
    # Visualize a grid and its heatmap
    sample_grid = torch.rand((128, 128, 3)).numpy()  # Replace with actual grid if needed
    sample_heatmap = torch.rand((128, 128)).numpy()  # Replace with actual heatmap if needed
    plot_heatmap(sample_grid, sample_heatmap)

if __name__ == "__main__":
    main()

