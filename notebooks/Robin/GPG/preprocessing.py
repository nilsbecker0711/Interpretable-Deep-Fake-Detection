
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an image: resize and normalize.

    Args:
        image_path (str): Path to the image.
        target_size (tuple): Desired output dimensions (height, width).

    Returns:
        np.ndarray: Preprocessed image array.
    """
    with Image.open(image_path) as img:
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array
