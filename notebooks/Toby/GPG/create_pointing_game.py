#1. 2x2 grids genereieren (1 fake)
#2. das als bild an bcos übergeben
#3. schau das prediction an
import os
import numpy as np
import random
from PIL import Image

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

def create_2x2_grids(preprocessed_dir, output_dir, grid_size=(2, 2)):
    """
    Create 2x2 grids with one fake image and the rest real based on preprocessed outputs.

    Args:
        preprocessed_dir (str): Path to the preprocessed directory containing 'real' and 'fake' subfolders.
        output_dir (str): Path to save the generated grids.
        grid_size (tuple): Dimensions of the grid (default: 2x2).
    """
    # Define real and fake image directories
    real_dir = os.path.join(preprocessed_dir, "original_sequences")
    #if only looking for fakes of certain kind move the fake root directory down to the fake method instead of being at the top root
    fake_dir = os.path.join(preprocessed_dir, "manipulated_sequences")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all image paths
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)

    for i, fake_img_path in enumerate(fake_images):
        # Randomly sample 3 real images
        real_samples = random.sample(real_images, grid_size[0] * grid_size[1] - 1)

        # Add one fake image
        images = real_samples + [fake_img_path]
        random.shuffle(images)  # Shuffle the placement of the fake image

        # Create the 2x2 grid
        grid = []
        for row in range(grid_size[0]):
            row_images = []
            for col in range(grid_size[1]):
                idx = row * grid_size[1] + col
                img = Image.open(images[idx])  # Load .png image
                row_images.append(np.array(img))  # Convert to NumPy array
            grid.append(np.hstack(row_images))
        grid_image = np.vstack(grid)

        # Save the grid
        fake_index = images.index(fake_img_path)
        grid_name = f"grid_{i}_fake_{fake_index}.npy"  # Save fake position in filename
        output_path = os.path.join(output_dir, grid_name)
        np.save(output_path, grid_image)  # Save as NumPy array for efficient loading
        
    print(f"Grids saved in {output_dir}")


if __name__ == "__main__":
    preprocessed_dir = "/Users/toby/Interpretable-Deep-Fake-Detection/datasets/FaceForensics++"  # Preprocessed root directory with 'frames/real' and 'frames/fake'
    output_dir = "/Users/toby/Interpretable-Deep-Fake-Detection/datasets/2x2_images"  # Directory to save 2x2 grids
    create_2x2_grids(preprocessed_dir, output_dir)