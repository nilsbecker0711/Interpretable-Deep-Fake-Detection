from data_handling import get_all_png_files
from grid_creation import create_2x2_grids
from preprocessing import preprocess_image
from evaluation import evaluate_bcos_resnet50
from visualization import plot_heatmap
import sys
#i needed to append my analysis path to system to access the bcos saved models
sys.path.append('/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/analysis')
import torch
import os

def main():
    # Step 1: Define Paths
    real_dir = "datasets/FaceForensics++/original_sequences/actors/c40/frames"
    fake_dir = "datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"
    output_dir = "datasets/2x2_grids"
    model_path = "../../weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth"

    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Retrieve Data
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images.")

    # Step 3: Create Grids
    create_2x2_grids(real_dir=real_dir, fake_dir=fake_dir, output_dir=output_dir)
    print(f"2x2 grids created and saved in {output_dir}.")

    # Step 4: Retrieve Grid Paths
    grid_paths = get_all_png_files(output_dir)
    print(f"Found {len(grid_paths)} grids for evaluation.")

    # Step 5: Preprocess Grids
    preprocessed_grids = [preprocess_image(grid_path) for grid_path in grid_paths]

    # Step 6: Load Model and Evaluate
    model = torch.load(model_path)
    metrics = evaluate_bcos_resnet50(model, preprocessed_grids)
    print(f"Evaluation Metrics: {metrics}")

    # Step 7: Visualize Heatmaps
    if preprocessed_grids and metrics.get("heatmaps"):
        sample_grid = preprocessed_grids[0].permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        sample_heatmap = metrics["heatmaps"][0]  # Select the first heatmap
        plot_heatmap(sample_grid, sample_heatmap)
    else:
        print("No heatmaps available for visualization.")

if __name__ == "__main__":
    main()


# conda env create -f deep_fake_env.yml
# conda activate deep_fake_env
# python GPG_pipeline.py
