# check if better 
import os
import random
import numpy as np
from PIL import Image

def create_2x2_grids(real_dir, fake_dir, output_dir, grid_size=(2, 2), max_grids=20):
    """
    Create 2x2 grids with one fake image and the rest real.

    Args:
        real_dir (str): Path to the directory with real images.
        fake_dir (str): Path to the directory with fake images.
        output_dir (str): Directory to save the generated grids.
        grid_size (tuple): Dimensions of the grid, default (2, 2).
        max_grids (int): Maximum number of grids to create.
    """
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
