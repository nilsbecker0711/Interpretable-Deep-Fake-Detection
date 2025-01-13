# check if better 
import torch
import torch.nn.functional as F

def evaluate_bcos_resnet50(model, preprocessed_grids):
    """
    Evaluate grids using BCos ResNet50 model.

    Args:
        model (torch.nn.Module): Pre-trained BCos ResNet50 model.
        preprocessed_grids (list): List of preprocessed grids (tensors).

    Returns:
        dict: Evaluation metrics including accuracy and heatmaps.
    """
    model.eval()
    all_predictions = []
    all_heatmaps = []

    for grid in preprocessed_grids:
        grid = grid.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(grid)
            probabilities = F.softmax(outputs, dim=1)
            prediction = probabilities.argmax(dim=1).item()

            all_predictions.append(prediction)
            all_heatmaps.append(outputs.cpu().numpy())  # Replace with actual heatmap logic if needed

    return {
        "accuracy": sum(all_predictions) / len(all_predictions),
        "heatmaps": all_heatmaps
    }
