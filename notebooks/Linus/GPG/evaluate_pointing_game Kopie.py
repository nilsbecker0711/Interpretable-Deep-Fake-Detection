import os
import numpy as np
from PIL import Image

def evaluate_heatmaps_with_comparison(heatmap_dir, grid_dir, grid_size=(2, 2)):
    """
    Evaluate heatmaps and compare the detected grid with the fake grid position.

    Args:
        heatmap_dir (str): Path to the directory containing heatmap images.
        grid_dir (str): Path to the directory containing grid files with fake positions in filenames.
        grid_size (tuple): Dimensions of the grid (default: 2x2).
    
    Returns:
        results (list): A list of dictionaries containing evaluation results for each heatmap.
    """
    heatmaps = [os.path.join(heatmap_dir, img) for img in os.listdir(heatmap_dir) if img.endswith('.png')]
    results = []

    for heatmap_path in heatmaps:
        # Extract the grid filename to find the corresponding grid file
        heatmap_name = os.path.basename(heatmap_path)
        grid_name = heatmap_name.replace('.png', '.npy')  # Assuming grid filenames match heatmap filenames
        grid_path = os.path.join(grid_dir, grid_name)

        if not os.path.exists(grid_path):
            print(f"Warning: No matching grid found for heatmap {heatmap_name}")
            continue

        # Load heatmap
        heatmap = Image.open(heatmap_path).convert("L")  # Convert to grayscale
        heatmap = np.array(heatmap)  # Convert to NumPy array

        # Split the heatmap into grid squares
        h, w = heatmap.shape
        square_h, square_w = h // grid_size[0], w // grid_size[1]

        min_white_pixels = float("inf")
        min_square = None

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Extract the square
                square = heatmap[i * square_h:(i + 1) * square_h, j * square_w:(j + 1) * square_w]
                
                # Count white pixels (assume white = 255)
                white_pixel_count = np.sum(square == 255)
                
                if white_pixel_count < min_white_pixels:
                    min_white_pixels = white_pixel_count
                    min_square = (i, j)

        # Extract the fake square from the grid filename
        fake_index = int(grid_name.split("_fake_")[-1].split(".")[0])
        fake_square = (fake_index // grid_size[1], fake_index % grid_size[1])

        # Compare detected square with the fake square
        correct = min_square == fake_square

        # Save result for this heatmap
        results.append({
            "heatmap": heatmap_path,
            "detected_square": min_square,
            "fake_square": fake_square,
            "correct": correct,
            "min_white_pixels": min_white_pixels,
        })

    return results


if __name__ == "__main__":
    heatmap_dir = "path_to_heatmaps"  # Directory where heatmap images are saved
    grid_dir = "path_to_grids"        # Directory where 2x2 grid files are saved
    grid_size = (2, 2)  # 2x2 grids

    # Evaluate heatmaps with comparison
    results = evaluate_heatmaps_with_comparison(heatmap_dir, grid_dir, grid_size)

    # Print results
    correct_count = 0
    for result in results:
        print(f"Heatmap: {result['heatmap']}")
        print(f"Detected square: {result['detected_square']}")
        print(f"Fake square: {result['fake_square']}")
        print(f"Correct: {result['correct']}")
        print(f"White pixels in detected square: {result['min_white_pixels']}\n")
        if result['correct']:
            correct_count += 1

    # Print overall accuracy
    accuracy = correct_count / len(results) if results else 0
    print(f"Overall Accuracy: {accuracy:.2f}")