#1. 2x2 grids genereieren (1 fake)
#2. das als bild an bcos übergeben
#3. schau das prediction an
import os
import random
import numpy as np
from PIL import Image

def create_2x2_grids(real_dir, fake_dir, output_dir, grid_size=(2, 2)):
    """
    Create 2x2 grids with one fake image and the rest real.

    Args:
        real_dir (str): Path to real images.
        fake_dir (str): Path to fake images.
        output_dir (str): Path to save the grids.
        grid_size (tuple): Dimensions of the grid (default: 2x2).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all real and fake image paths
        # Traverse through the subdirectories of real_dir to get all real image paths
    real_images = []
    for root, dirs, files in os.walk(real_dir):
        for file in files:
            if file.endswith('.png'):  # Look for .png files
                real_images.append(os.path.join(root, file))

    # Traverse through the subdirectories of fake_dir to get all fake image paths
    fake_images = []
    for root, dirs, files in os.walk(fake_dir):
        for file in files:
            if file.endswith('.png'):  # Look for .png files
                fake_images.append(os.path.join(root, file))

    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")

    # Loop through fake images and create grids
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
                img = Image.open(images[idx])  # Load the image
                row_images.append(np.array(img))  # Convert to NumPy array
            grid.append(np.hstack(row_images))
        grid_image = np.vstack(grid)

        # Save the grid image
        fake_index = images.index(fake_img_path)
        grid_name = f"grid_{i}_fake_{fake_index}.npy"  # Save fake position in filename
        output_path = os.path.join(output_dir, grid_name)
        np.save(output_path, grid_image)  # Save as NumPy array

    print(f"Grids saved in {output_dir}")

if __name__ == "__main__":
    real_dir = "datasets/FaceForensics++/original_sequences/actors/c40/frames"  # Path to real images
    fake_dir = "datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"  # Path to fake images
    output_dir = "datasets/2x2_images"  # Output path for saving 2x2 grids

    create_2x2_grids(real_dir, fake_dir, output_dir)

