import os
import argparse
import numpy as np
from PIL import Image
import random

def get_all_png_files(directory):
    """
    Retrieve all PNG file paths from a directory.
    """
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])

def create_difference_masks(real_dir, fake_dir, output_dir, max_masks=500, threshold=30):
    """
    Create masks by computing pixel-wise differences between matching real and fake images.
    
    Args:
        real_dir (str): Path to real images.
        fake_dir (str): Path to fake images.
        output_dir (str): Directory where difference masks will be saved.
        max_masks (int): Maximum number of masks to generate.
        threshold (int): Threshold for binary difference mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted lists of images
    real_images = get_all_png_files(real_dir)
    fake_images = get_all_png_files(fake_dir)
    
    # Ensure same number of images are in both folders
    common_length = min(len(real_images), len(fake_images))
    if common_length == 0:
        raise ValueError("No matching images found in real and fake directories.")
    
    # Limit the number of processed image pairs
    image_pairs = list(zip(real_images[:common_length], fake_images[:common_length]))
    if len(image_pairs) > max_masks:
        image_pairs = random.sample(image_pairs, max_masks)
    
    print(f"Processing {len(image_pairs)} real-fake image pairs...")
    
    for idx, (real_path, fake_path) in enumerate(image_pairs):
        with Image.open(real_path).convert('L') as real_img, Image.open(fake_path).convert('L') as fake_img:
            real_array = np.array(real_img, dtype=np.float32)
            fake_array = np.array(fake_img, dtype=np.float32)
            
            # Compute absolute difference
            diff_mask = np.abs(real_array - fake_array)
            
            # Apply threshold to highlight differences
            binary_mask = (diff_mask > threshold) * 255
            
            # Save mask
            mask_image = Image.fromarray(binary_mask.astype(np.uint8))
            mask_filename = f"mask_{idx}.png"
            mask_image.save(os.path.join(output_dir, mask_filename))
    
    print(f"{len(image_pairs)} masks saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate difference masks from real and fake images.")
    parser.add_argument("--real_dir", type=str, default="datasets/FaceForensics++/original_sequences/actors/c40/frames", help="Path to real images.")
    parser.add_argument("--fake_dir", type=str, default="datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames", help="Path to fake images.")
    parser.add_argument("--output_dir", type=str, default="datasets/DMPG_grids", help="Directory to save difference masks.")
    parser.add_argument("--max_masks", type=int, default=3, help="Maximum number of masks to create.")
    parser.add_argument("--threshold", type=int, default=30, help="Threshold for binary difference mask.")
    
    args = parser.parse_args()
    create_difference_masks(args.real_dir, args.fake_dir, args.output_dir, args.max_masks, args.threshold)

if __name__ == "__main__":
    main()


#python notebooks/Linus/DifferenceMaskPointingGame/DMPG_creation.py --real_dir "/path/to/real_images" --fake_dir "/path/to/fake_images" --output_dir "/path/to/masks" --threshold 30