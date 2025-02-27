import os
import argparse
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def get_all_png_files(root_folder, filter_keyword=None):
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

def create_GPG_grids(real_dir, fake_dir, base_output_dir, grid_size=(3, 3), max_grids=2):
    print(f"[DEBUG] create_GPG_grids - real_dir={real_dir}, fake_dir={fake_dir}, base_output_dir={base_output_dir}")
    
    # Create output directories for 3-channel and 6-channel tensors.
    output_dir_3ch = os.path.join(base_output_dir, "3ch")
    output_dir_6ch = os.path.join(base_output_dir, "6ch")
    os.makedirs(output_dir_3ch, exist_ok=True)
    os.makedirs(output_dir_6ch, exist_ok=True)
    
    # Get image paths.
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)

    print(f"[DEBUG] Real images found: {len(real_images)}")
    print(f"[DEBUG] Fake images found: {len(fake_images)}")
    print(f"[DEBUG] Output dir: {base_output_dir}")

    n_imgs = int(grid_size[0]) * int(grid_size[1])
    grid_count = 0
    side = int(np.sqrt(n_imgs))  # side length of the grid (e.g., 3 for a 3x3 grid)

    for fake_img_path in fake_images:
        if grid_count >= max_grids:
            break

        print(f"[DEBUG] Creating grid {grid_count} using fake image: {fake_img_path}")
        
        needed_real = n_imgs - 1
        if len(real_images) < needed_real:
            print(f"[DEBUG] Not enough real images (need {needed_real}), breaking.")
            break

        # Randomly select the required number of real images.
        real_samples = random.sample(real_images, needed_real)
        # Combine with the fake image.
        images = real_samples + [fake_img_path]
        random.shuffle(images)
  
        # Load images and convert to tensor.
        transform = T.ToTensor()
        png_tensors = []
        for img_path in images:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # ensure 3-channel RGB
                png_tensors.append(transform(img))

        print(f"[DEBUG] Number of loaded PNG tensors: {len(png_tensors)}")

        # Stack into a tensor of shape [n_imgs, C, H, W].
        stacked = torch.stack(png_tensors, dim=0)

        # Use your provided transformation to create the grid.
        grid_tensor = stacked.view(-1, side, side, *stacked.shape[-3:]) \
                             .permute(0, 3, 2, 4, 1, 5) \
                             .reshape(-1,
                                      stacked.shape[1],
                                      stacked.shape[2] * side,
                                      stacked.shape[3] * side)

        # Determine fake image's position in the grid for naming.
        fake_index = images.index(fake_img_path)
        base_name = f"grid_{grid_count}_fake_{fake_index}.pt"

        # Save the 3-channel grid tensor.
        path_3ch = os.path.join(output_dir_3ch, base_name)
        torch.save(grid_tensor, path_3ch)
        print(f"[DEBUG] Saved 3-channel grid tensor to: {path_3ch}")

        # Create the 6-channel version by concatenating the grid and its inverse.
        six_ch_tensor = torch.cat([grid_tensor, 1.0 - grid_tensor], dim=0)  # shape: [6, H, W]
        base_name_6ch = f"grid_{grid_count}_fake_{fake_index}.pt"
        path_6ch = os.path.join(output_dir_6ch, base_name_6ch)
        torch.save(six_ch_tensor, path_6ch)
        print(f"[DEBUG] Saved 6-channel grid tensor to: {path_6ch}")

        grid_count += 1

    print("[DEBUG] Grid creation complete.")

def main(real_dir, fake_dir, base_output_dir, grid_size, max_grids):
    print("[DEBUG] main - Parameters:")
    print(f"        real_dir={real_dir}")
    print(f"        fake_dir={fake_dir}")
    print(f"        base_output_dir={base_output_dir}")
    os.makedirs(base_output_dir, exist_ok=True)
    create_GPG_grids(real_dir, fake_dir, base_output_dir, grid_size, max_grids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation grids with resizing.")
    parser.add_argument("--real_dir", type=str, default="datasets/FaceForensics++/original_sequences/actors/c40/frames", help="Directory containing real images.")
    parser.add_argument("--fake_dir", type=str, default="datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames", help="Directory containing fake images.")
    parser.add_argument("--base_output_dir", type=str, default="datasets/GPG_grids", help="Base output directory to save grids.")
    parser.add_argument("--grid_rows", type=int, default=3, help="Number of rows in the grid.")
    parser.add_argument("--grid_cols", type=int, default=3, help="Number of columns in the grid.")
    parser.add_argument("--max_grids", type=int, default=2, help="Maximum number of grids to create.")
    
    args = parser.parse_args()
    grid_size_tuple = (args.grid_rows, args.grid_cols)
    main(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        base_output_dir=args.base_output_dir,
        grid_size=grid_size_tuple,
        max_grids=args.max_grids
    )
