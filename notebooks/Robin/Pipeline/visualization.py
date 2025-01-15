
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(image, heatmap):
    """
    Overlay a heatmap on the image.

    Args:
        image (np.ndarray): Original image.
        heatmap (np.ndarray): Heatmap values.
    """
    plt.figure()
    plt.imshow(image, alpha=0.6)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.colorbar()
    plt.show()


def visualize_predictions(grid, predictions):
    """
    Visualize the predictions on a grid.

    Args:
        grid (np.ndarray): The grid of images.
        predictions (list): Predictions for each image in the grid.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    for i, pred in enumerate(predictions):
        plt.text((i % 2) * grid.shape[1] // 2, (i // 2) * grid.shape[0] // 2, str(pred), 
                 color='red', fontsize=12, ha='center', va='center')
    plt.show()
