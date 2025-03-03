
import os
import random
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

# Add analysis path for BCos ResNet50 models
sys.path.append('/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/analysis')
from b_cos.resnet import resnet50

class GridCreator:
    """Handles the creation of 2x2 grids from real and fake images."""

    @staticmethod
    def create_2x2_grids(real_dir, fake_dir, output_dir, grid_size=(2, 2), max_grids=20):
        os.makedirs(output_dir, exist_ok=True)

        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.png')]
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.png')]

        for i, fake_img_path in enumerate(fake_images[:max_grids]):
            real_samples = random.sample(real_images, grid_size[0] * grid_size[1] - 1)
            images = real_samples + [fake_img_path]
            random.shuffle(images)

            grid = []
            for row in range(grid_size[0]):
                row_images = [np.array(Image.open(images[col])) for col in range(row * grid_size[1], (row + 1) * grid_size[1])]
                grid.append(np.hstack(row_images))
            grid_image = np.vstack(grid)

            grid_name = f"grid_{i}.png"
            grid_path = os.path.join(output_dir, grid_name)
            Image.fromarray(grid_image).save(grid_path)
        print(f"Grids saved in {output_dir}")

class ModelEvaluator:
    """Handles evaluation of grids using a given model."""

    def __init__(self, model, preprocess_func):
        self.model = model
        self.preprocess_func = preprocess_func

    def evaluate(self, grid_paths):
        """Evaluate grids and extract heatmaps for visualization."""
        self.model.eval()
        all_scores = []
        all_heatmaps = []

        for grid_path in grid_paths:
            grid_tensor = self.preprocess_func(grid_path).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = self.model(grid_tensor)
                probabilities = F.softmax(output, dim=1)
                prediction = probabilities.argmax(dim=1).item()
                all_scores.append(prediction)
                all_heatmaps.append(probabilities.cpu().numpy())

        avg_score = np.mean(all_scores)
        return {"average_score": avg_score, "total_images": len(grid_paths), "heatmaps": all_heatmaps}

    def visualize_results(self, grid_path, heatmap, accuracy):
        """Visualize the grid with heatmap and accuracy information."""
        grid_image = np.array(Image.open(grid_path))

        plt.figure(figsize=(8, 8))
        plt.imshow(grid_image, alpha=0.6)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.colorbar()
        plt.title(f"Accuracy: {accuracy:.2f}")
        plt.show()

class ModelAndDataManager:
    """Manages model loading and dataset paths."""

    def __init__(self, model_path, preprocess_func):
        self.model_path = model_path
        self.preprocess_func = preprocess_func

    def load_model(self):
        """Loads the model from the specified path and maps it to the CPU."""
        return torch.load(self.model_path, map_location=torch.device('cpu'))

    def get_grid_paths(self, output_dir):
        """Retrieves all grid paths in the output directory."""
        return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]

    @staticmethod
    def get_all_png_files(root_folder):
        """
        Recursively collects all .png file paths from a given root folder.

        Args:
            root_folder (str): Path to the root folder to search.

        Returns:
            list: List of paths to all .png files found within the folder hierarchy.
        """
        png_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            for file in filenames:
                if file.endswith('.png') and 'frames' in dirpath:
                    png_files.append(os.path.join(dirpath, file))
        return png_files

# Example usage
if __name__ == "__main__":
 # Paths (Update these as necessary)
    real_dir = "/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/datasets/FaceForensics++/original_sequences/actors/c40/frames"
    fake_dir = "/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"
    output_dir = "/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/datasets/2x2_grids"
    model_path = "/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-1/weights/B_cos/ResNet50/b_cos_model_1732594597.04.pth"

    # Step 1: Retrieve image files
    print("Retrieving image files...")
    all_real_images = ModelAndDataManager.get_all_png_files(real_dir)
    all_fake_images = ModelAndDataManager.get_all_png_files(fake_dir)
    print(f"Found {len(all_real_images)} real images and {len(all_fake_images)} fake images.")

    # Step 2: Create grids
    print("Creating 2x2 grids...")
    GridCreator.create_2x2_grids(real_dir, fake_dir, output_dir)

    # Step 3: Retrieve generated grids
    print("Retrieving grid paths...")
    manager = ModelAndDataManager(model_path, preprocess_func=lambda x: torch.tensor(np.array(Image.open(x)) / 255.0).permute(2, 0, 1))
    grid_paths = manager.get_grid_paths(output_dir)
    print(f"Found {len(grid_paths)} grids for evaluation.")

    # Step 4: Load model
    print("Loading model...")
    model = manager.load_model()

    # Step 5: Evaluate grids
    print("Evaluating grids...")
    evaluator = ModelEvaluator(model, manager.preprocess_func)
    metrics = evaluator.evaluate(grid_paths)
    print(f"Evaluation Metrics: {metrics}")

    # Step 6: Visualize a sample grid
    print("Visualizing a sample grid...")
    if grid_paths and metrics["heatmaps"]:
        sample_grid = grid_paths[0]
        sample_heatmap = metrics["heatmaps"][0][0]  # Replace [0][0] with proper heatmap indexing if needed
        evaluator.visualize_results(sample_grid, sample_heatmap, metrics["average_score"])
