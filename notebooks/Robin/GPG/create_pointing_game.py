#1. 2x2 grids genereieren (1 fake)
#2. das als bild an bcos übergeben
#3. schau das prediction an
import os
import numpy as np
import random
from PIL import Image

def create_2x2_grids(preprocessed_dir, output_dir, grid_size=(2, 2)):
    """
    Create 2x2 grids with one fake image and the rest real based on preprocessed outputs.

    Args:
        preprocessed_dir (str): Path to the preprocessed directory containing 'real' and 'fake' subfolders.
        output_dir (str): Path to save the generated grids.
        grid_size (tuple): Dimensions of the grid (default: 2x2).
    """
    # Define real and fake image directories
    real_dir = os.path.join(preprocessed_dir, "frames", "real")
    fake_dir = os.path.join(preprocessed_dir, "frames", "fake")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all image paths
    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith('.png')]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith('.png')]

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
    preprocessed_dir = "path_to_preprocessed_data"  # Preprocessed root directory with 'frames/real' and 'frames/fake'
    output_dir = "path_to_grids"  # Directory to save 2x2 grids
    create_2x2_grids(preprocessed_dir, output_dir)