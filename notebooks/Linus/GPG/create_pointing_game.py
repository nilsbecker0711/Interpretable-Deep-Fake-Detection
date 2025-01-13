import os
import numpy as np
import random
from PIL import Image

class GridGenerator:
    def __init__(self, real_dir, fake_dir, output_dir, grid_size=(2, 2), max_grids=20):
        """
        Initializes the GridGenerator class.

        Args:
            real_dir (str): Path to the directory with real images.
            fake_dir (str): Path to the directory with fake images.
            output_dir (str): Path to save the generated grids.
            grid_size (tuple): Dimensions of the grid (default: 2x2).
            max_grids (int): Maximum number of grids to generate.
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.output_dir = output_dir
        self.grid_size = grid_size
        self.max_grids = max_grids

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_all_png_files(self, root_folder):
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

    def create_2x2_grids(self):
        """
        Creates 2x2 grids with one fake image and the rest real images, and saves them to the output directory.
        """
        # Collect all image paths
        real_images = self.get_all_png_files(self.real_dir)
        fake_images = self.get_all_png_files(self.fake_dir)
        
        print(f"Real images found: {len(real_images)}")
        print(f"Fake images found: {len(fake_images)}")
        
        print("Real Directory Absolute Path:", os.path.abspath(self.real_dir))
        print("Fake Directory Absolute Path:", os.path.abspath(self.fake_dir))
        
        grid_count = 0

        for i, fake_img_path in enumerate(fake_images):
            if grid_count >= self.max_grids:
                break
            # Randomly sample 3 real images
            real_samples = random.sample(real_images, self.grid_size[0] * self.grid_size[1] - 1)

            # Add one fake image
            images = real_samples + [fake_img_path]
            random.shuffle(images)  # Shuffle the placement of the fake image

            # Create the 2x2 grid
            grid = []
            for row in range(self.grid_size[0]):
                row_images = []
                for col in range(self.grid_size[1]):
                    idx = row * self.grid_size[1] + col
                    img = Image.open(images[idx])  # Load .png image
                    row_images.append(np.array(img))  # Convert to NumPy array
                grid.append(np.hstack(row_images))
            grid_image = np.vstack(grid)

            # Save the grid as a PNG image
            fake_index = images.index(fake_img_path)
            grid_name = f"grid_{i}_fake_{fake_index}.png"  # Use .png as the file extension
            output_path = os.path.join(self.output_dir, grid_name)

            # Convert NumPy array to PIL Image and save
            grid_image = Image.fromarray(grid_image.astype(np.uint8))  # Ensure data is in uint8 format
            grid_image.save(output_path)  # Save as a PNG file
            
            grid_count += 1
        print(f"Grids saved in {self.output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize the GridGenerator class
    generator = GridGenerator(
        real_dir="/Interpretable-Deep-Fake-Detection/datasets/FaceForensics++/original_sequences/actors/c40/frames",
        fake_dir="/Interpretable-Deep-Fake-Detection/datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames",
        output_dir="/Interpretable-Deep-Fake-Detection/datasets/2x2_images",
        grid_size=(2, 2),
        max_grids=20
    )
    # Create grids
    generator.create_2x2_grids()