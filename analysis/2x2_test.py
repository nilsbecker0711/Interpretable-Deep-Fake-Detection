import numpy as np
import matplotlib.pyplot as plt

def load_and_display_npy_image(npy_path):
    """
    Load a .npy image (NumPy array) and display it.
    
    Args:
        npy_path (str): Path to the .npy file containing the image data.
    """
    # Load the .npy file
    grid_image = np.load(npy_path)

    # Display the grid image using matplotlib
    plt.imshow(grid_image)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

if __name__ == "__main__":
    npy_path = "datasets/2x2_images/grid_1_fake_2.npy"  # Replace with the actual path to your .npy file
    load_and_display_npy_image(npy_path)