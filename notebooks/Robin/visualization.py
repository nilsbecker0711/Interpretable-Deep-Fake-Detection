#check if better 
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(image, heatmap):
    """
    Overlay a heatmap on the image.

    Args:
        image (np.ndarray): Original image.
        heatmap (np.ndarray): Heatmap values.
    """
    plt.imshow(image, alpha=0.6)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.colorbar()
    plt.show()
