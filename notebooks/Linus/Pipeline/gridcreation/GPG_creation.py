import os
import argparse
import random
import numpy as np
from PIL import Image

def get_all_png_files(root_folder, filter_keyword=None):
    """
    Recursively collects all .png file paths from a given root folder.

    Args:
        root_folder (str): Path to the root folder to search.
        filter_keyword (str, optional): Only include paths with this keyword. Defaults to None.

    Returns:
        list: List of paths to all .png files found within the folder hierarchy.
    """
    png_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        if filter_keyword and filter_keyword not in dirpath:
            continue
        for file in filenames:
            if file.endswith('.png'):
                png_files.append(os.path.join(dirpath, file))
    return png_files

def create_2x2_grids(real_dir, fake_dir, base_output_dir, grid_size=(3, 3), max_grids=500, target_grid_size=(224,224)):
    """
    Create grids with one fake image and the rest real.
    For each generated grid, resize it to the target_grid_size and then save both:
      - A 3-channel version (saved as a PNG) in a subfolder "3ch"
      - A 6-channel version (RGB + inverse, saved as a NumPy file) in a subfolder "6ch"
    
    Args:
        real_dir (str): Directory with real images.
        fake_dir (str): Directory with fake images.
        base_output_dir (str): Base directory to save the grids.
        grid_size (tuple): Dimensions of the grid (e.g., (3, 3)).
        max_grids (int): Maximum number of grids to create.
        target_grid_size (tuple): Desired final size of the grid (height, width).
    """
    # Create subfolders for 3-channel and 6-channel outputs.
    output_dir_3ch = os.path.join(base_output_dir, "3ch")
    output_dir_6ch = os.path.join(base_output_dir, "6ch")
    os.makedirs(output_dir_3ch, exist_ok=True)
    os.makedirs(output_dir_6ch, exist_ok=True)
    
    # Set a fixed seed for reproducibility.
    random.seed(42)
    
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)
    
    required_real = grid_size[0] * grid_size[1] - 1
    if len(real_images) < required_real:
        raise ValueError("Not enough real images to create grids.")
    if len(fake_images) == 0:
        raise ValueError("No fake images found in the fake images directory.")
    
    grid_count = 0
    for fake_img_path in fake_images:
        if grid_count >= max_grids:
            break
        
        # Randomly sample real images and add the fake image.
        real_samples = random.sample(real_images, required_real)
        images = real_samples + [fake_img_path]
        random.shuffle(images)  # Shuffle once per grid.
        
        # Build the grid by horizontally stacking each row and then vertically stacking the rows.
        grid_rows = []
        for row in range(grid_size[0]):
            row_images = []
            for col in range(grid_size[1]):
                idx = row * grid_size[1] + col
                with Image.open(images[idx]) as img:
                    img = img.convert('RGB')
                    row_images.append(np.array(img))
            grid_rows.append(np.hstack(row_images))
        grid_image = np.vstack(grid_rows)  # This is the base 3-channel grid.
        
        # Resize the grid to the desired target_grid_size.
        current_size = grid_image.shape[:2]  # (height, width)
        if current_size != target_grid_size:
            pil_grid = Image.fromarray(grid_image.astype(np.uint8))
            # Note: PIL expects size as (width, height)
            pil_grid = pil_grid.resize((target_grid_size[1], target_grid_size[0]), Image.LANCZOS)
            grid_image = np.array(pil_grid)
        
        # Determine fake image position for naming.
        fake_index = images.index(fake_img_path)
        base_name = f"grid_{grid_count}_fake_{fake_index}"
        
        # Save the 3-channel version.
        path_3ch = os.path.join(output_dir_3ch, base_name + ".png")
        Image.fromarray(grid_image.astype(np.uint8)).save(path_3ch)
        
        # Create the 6-channel version:
        # Normalize to [0,1], compute the inverse, and then concatenate.
        grid_float = grid_image.astype(np.float32) / 255.0
        inverse = 1 - grid_float
        grid_image_6 = np.concatenate([grid_float, inverse], axis=-1)
        path_6ch = os.path.join(output_dir_6ch, base_name + ".npy")
        np.save(path_6ch, grid_image_6)
        
        grid_count += 1
    
    print(f"{grid_count} grids saved in {base_output_dir}")








    

def main(real_dir, fake_dir, base_output_dir, grid_size, max_grids, target_grid_size):
    # Ensure the base output directory exists.
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create grids using the updated function that performs resizing.
    create_2x2_grids(real_dir, fake_dir, base_output_dir, grid_size, max_grids, target_grid_size)
    
    print("Grid creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation grids with resizing.")
    parser.add_argument(
        "--real_dir",
        type=str,
        default="datasets/FaceForensics++/original_sequences/actors/c40/frames",
        help="Directory containing real images."
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames",
        help="Directory containing fake images."
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="datasets/2x2_grids",
        help="Base output directory to save grids."
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=3,
        help="Number of rows in the grid."
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns in the grid."
    )
    parser.add_argument(
        "--max_grids",
        type=int,
        default=500,
        help="Maximum number of grids to create."
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=224,
        help="Target height for the grid image."
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=224,
        help="Target width for the grid image."
    )
    
    args = parser.parse_args()
    
    # Create tuples for grid dimensions and target size.
    grid_size_tuple = (args.grid_rows, args.grid_cols)
    target_grid_size_tuple = (args.target_height, args.target_width)
    
    main(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        base_output_dir=args.base_output_dir,
        grid_size=grid_size_tuple,
        max_grids=args.max_grids,
        target_grid_size=target_grid_size_tuple
    )


# python notebooks/Linus/Pipeline/gridcreation/GPG_creation.py --grid_rows 3 --grid_cols 3 --max_grid --real_dir "/new/real/path" --fake_dir "/new/fake/path" --base_output_dir "/output/path" --max_grids 100 --target_height 224 --target_width 224