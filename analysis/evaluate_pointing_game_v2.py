import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
#load bcos model
from b_cos.bcosconv2d import BcosConv2d
from b_cos.resnet import resnet50

def load_bcos_model(model_path = "models/b_cos/b_cos_model_x.pth"):
    """
    Load the BCos model from a given path.

    Args:
        model_path (str): Path to the model file.
    """
    model = resnet50(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
  # Ensure this matches the model architecture
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_bcos_model()

print("Model Parameters:")
for name, param in model.named_parameters():
    print(f"Name: {name}")
    print(f"Value: {param}")
    print(f"Shape: {param.shape}\n")

# To only print parameter names
print("Parameter Names:")
for name, _ in model.named_parameters():
    print(name)
    
def load_grid_images(grid_dir = "datasets/2x2_images"):
    grid_images = []
    for file_name in os.listdir(grid_dir):
        if file_name.endswith('.npy'):
            grid_image_path = os.path.join(grid_dir, file_name)
            grid_images.append(np.load(grid_image_path))  # Load the .npy image
    return grid_images

""" # Generate saliency map using the model
def generate_saliency_map(model, image, device):
    image = torch.tensor(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    image = image.float()  # Ensure the image is a float tensor

    image.requires_grad = True  # Enable gradient computation for the image

    # Forward pass
    output = model(image)

    # Compute the loss (for simplicity, assume binary classification, adjust if needed)
    loss = F.cross_entropy(output, torch.tensor([1]).to(device))  # Example with target class 1
    model.zero_grad()
    loss.backward()  # Backpropagate to get gradients of the image

    saliency_map = image.grad.data.abs().squeeze(0).cpu().numpy()  # Take absolute gradient
    return saliency_map

# Perform pointing game evaluation
def perform_pointing_game(saliency_maps, grid_dir, grid_size=(2, 2)):
    results = []

    for idx, saliency_map in enumerate(saliency_maps):
        # Load the corresponding grid image filename (assumes grid filename matches)
        grid_name = os.listdir(grid_dir)[idx]
        fake_index = int(grid_name.split("_fake_")[-1].split(".")[0])  # Extract fake position from filename

        # Split the saliency map into grid squares (2x2)
        h, w = saliency_map.shape
        square_h, square_w = h // grid_size[0], w // grid_size[1]

        # Find the square with the highest activation in the saliency map
        max_activation = -1
        max_square = None
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                square = saliency_map[i * square_h:(i + 1) * square_h, j * square_w:(j + 1) * square_w]
                non_zero_pixels = np.sum(square > 0)  # Count non-zero (white) pixels
                if non_zero_pixels > max_activation:
                    max_activation = non_zero_pixels
                    max_square = (i, j)

        # Compare detected square with the fake square
        fake_square = (fake_index // grid_size[1], fake_index % grid_size[1])

        # Store the result (True if detected correctly, False otherwise)
        correct = max_square == fake_square
        results.append({
            "grid_name": grid_name,
            "detected_square": max_square,
            "fake_square": fake_square,
            "correct": correct,
            "max_activation": max_activation,
        })

        # Visualize saliency map and grid
        plt.imshow(saliency_map)
        plt.title(f"Saliency Map for {grid_name}")
        plt.colorbar()
        plt.show()

    return results

# Main function to load model, generate saliency maps, and perform the pointing game
def main():
    model_path = "/Users/toby/Interpretable-Deep-Fake-Detection/models/b_cos/b_cos_model_x.pth"
    grid_dir = "datasets/2x2_images"  # Path to your generated 2x2 grid images
    output_dir = "analysis"  # Output path for saving results (optional)

    # Load the BCos ResNet model
    model = load_bcos_model(model_path)

    # Load the grid images
    grid_images = load_grid_images(grid_dir)

    # Set the device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to GPU if available

    # Generate saliency maps for each grid image
    saliency_maps = []
    for grid_image in grid_images:
        saliency_map = generate_saliency_map(model, grid_image, device)
        saliency_maps.append(saliency_map)

    # Perform the pointing game and display results
    results = perform_pointing_game(saliency_maps, grid_dir)

    # Print results
    correct_count = 0
    for result in results:
        print(f"Grid: {result['grid_name']}")
        print(f"Detected square: {result['detected_square']}")
        print(f"Fake square: {result['fake_square']}")
        print(f"Correct: {result['correct']}")
        print(f"Max activation (white pixels): {result['max_activation']}\n")
        if result['correct']:
            correct_count += 1

    # Print overall accuracy
    accuracy = correct_count / len(results) if results else 0
    print(f"Overall Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main() """