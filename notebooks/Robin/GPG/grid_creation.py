
import os
import random
from PIL import Image

def create_2x2_grids(preprocessed_dir, output_dir, grid_size=(2, 2)):
    """
    Create 2x2 grids with one fake image and the rest real based on preprocessed outputs.

    Args:
        preprocessed_dir (str): Path to the preprocessed directory containing 'real' and 'fake' subfolders.
        output_dir (str): Path to save the generated grids.
        grid_size (tuple): Dimensions of the grid, default (2, 2).
    """
    os.makedirs(output_dir, exist_ok=True)

    real_dir = os.path.join(preprocessed_dir, 'real')
    fake_dir = os.path.join(preprocessed_dir, 'fake')

    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('.png', '.jpg'))]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('.png', '.jpg'))]

    for idx, fake_path in enumerate(fake_images):
        fake_img = Image.open(fake_path)
        real_sample = random.sample(real_images, grid_size[0] * grid_size[1] - 1)

        grid_imgs = [fake_img] + [Image.open(real) for real in real_sample]
        random.shuffle(grid_imgs)

        grid_width, grid_height = grid_size
        single_width, single_height = grid_imgs[0].size
        grid = Image.new('RGB', (grid_width * single_width, grid_height * single_height))

        for pos, img in enumerate(grid_imgs):
            x = (pos % grid_width) * single_width
            y = (pos // grid_width) * single_height
            grid.paste(img, (x, y))

        grid.save(os.path.join(output_dir, f'grid_{idx}.jpg'))
