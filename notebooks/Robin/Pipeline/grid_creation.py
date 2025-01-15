
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
        grid_size (tuple): Dimensions of the grid, default (2x2).
        max_grids (int): Maximum number of grids to create.

    Raises:
        ValueError: If there aren't enough real images for grid creation.
    """
    os.makedirs(output_dir, exist_ok=True)
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)

    if len(real_images) < grid_size[0] * grid_size[1] - 1:
        raise ValueError("Not enough real images to create grids.")

    grid_count = 0
    for i, fake_img_path in enumerate(fake_images):
        if grid_count >= max_grids:
            break
        real_samples = random.sample(real_images, grid_size[0] * grid_size[1] - 1)
        images = real_samples + [fake_img_path]
        random.shuffle(images)

        grid = []
        for row in range(grid_size[0]):
            row_images = []
            for col in range(grid_size[1]):
                idx = row * grid_size[1] + col
                img = Image.open(images[idx])
                row_images.append(np.array(img))
            grid.append(np.hstack(row_images))
        grid_image = np.vstack(grid)

        grid_name = f"grid_{grid_count}_fake_{images.index(fake_img_path)}.png"
        grid_path = os.path.join(output_dir, grid_name)
        Image.fromarray(grid_image.astype(np.uint8)).save(grid_path)
        grid_count += 1

    print(f"Grids saved in {output_dir}")
