
import os

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
