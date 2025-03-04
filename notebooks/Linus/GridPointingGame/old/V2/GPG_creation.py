import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import yaml


def load_config(path, additional_args={}):
    # Parse primary config.
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Attempt to load training config.
    try:
        with open('./training/config/train_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
            config2 = yaml.safe_load(f)
    
    # If label_dict is present in the first config, update the training config.
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    
    config.update(config2)
    
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
        
    # Override with any additional arguments.
    for key, value in additional_args.items():
        config[key] = value
    return config

class GPGCreator:
    def __init__(self, real_dir, fake_dir, base_output_dir, grid_size=(3, 3), max_grids=2):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.base_output_dir = base_output_dir
        self.grid_size = grid_size
        self.max_grids = max_grids

        # Create output directories for 3-channel and 6-channel tensors.
        self.output_dir_3ch = os.path.join(self.base_output_dir, "3ch")
        self.output_dir_6ch = os.path.join(self.base_output_dir, "6ch")
        os.makedirs(self.output_dir_3ch, exist_ok=True)
        os.makedirs(self.output_dir_6ch, exist_ok=True)

    def get_all_png_files(self, root_folder, filter_keyword=None):
        """
        Recursively collects all .png file paths from a given root folder.
        """
        png_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            if filter_keyword and filter_keyword not in dirpath:
                continue
            for file in filenames:
                if file.endswith('.png'):
                    png_files.append(os.path.join(dirpath, file))
        return png_files

    def create_GPG_grids(self):
        print(f"[DEBUG] create_GPG_grids - real_dir={self.real_dir}, fake_dir={self.fake_dir}, base_output_dir={self.base_output_dir}")

        # Get image paths.
        real_images = self.get_all_png_files(self.real_dir)
        fake_images = self.get_all_png_files(self.fake_dir)

        print(f"[DEBUG] Real images found: {len(real_images)}")
        print(f"[DEBUG] Fake images found: {len(fake_images)}")

        n_imgs = int(self.grid_size[0]) * int(self.grid_size[1])
        grid_count = 0
        side = int(np.sqrt(n_imgs))  # Assumes a square grid.

        for fake_img_path in fake_images:
            if grid_count >= self.max_grids:
                break

            print(f"[DEBUG] Creating grid {grid_count} using fake image: {fake_img_path}")
            
            needed_real = n_imgs - 1
            if len(real_images) < needed_real:
                print(f"[DEBUG] Not enough real images (need {needed_real}), breaking.")
                break

            real_samples = random.sample(real_images, needed_real)
            images = real_samples + [fake_img_path]
            random.shuffle(images)

            fake_index = images.index(fake_img_path)
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            print(f"[DEBUG] Original fake index: {fake_index}, Final fake index: {final_fake_index}")
  
            # Load images and convert to tensor.
            transform = T.ToTensor()
            png_tensors = []
            for img_path in images:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    png_tensors.append(transform(img))

            print(f"[DEBUG] Number of loaded PNG tensors: {len(png_tensors)}")

            # Stack into a tensor of shape [n_imgs, C, H, W].
            stacked = torch.stack(png_tensors, dim=0)

            # Create the grid tensor.
            grid_tensor = stacked.view(-1, side, side, *stacked.shape[-3:]) \
                                 .permute(0, 3, 2, 4, 1, 5) \
                                 .reshape(-1,
                                          stacked.shape[1],
                                          stacked.shape[2] * side,
                                          stacked.shape[3] * side)

            base_name = f"grid_{grid_count}_fake_{final_fake_index}.pt"

            # Save the 3-channel grid tensor.
            path_3ch = os.path.join(self.output_dir_3ch, base_name)
            torch.save(grid_tensor, path_3ch)
            print(f"[DEBUG] Saved 3-channel grid tensor to: {path_3ch}")

            # Create the 6-channel version.
            six_ch_tensor = torch.cat([grid_tensor, 1.0 - grid_tensor], dim=1)
            path_6ch = os.path.join(self.output_dir_6ch, base_name)
            torch.save(six_ch_tensor, path_6ch)
            print(f"[DEBUG] Saved 6-channel grid tensor to: {path_6ch}")

            grid_count += 1

        print("[DEBUG] Grid creation complete.")