# check if better 
import os

def get_all_png_files(root_folder):
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
            if file.endswith('.png'):
                png_files.append(os.path.join(dirpath, file))
    return png_files
