# gradcam
import os
import sys 

# Set up project root and ensure it's in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from training.detectors.resnet34_detector import ResnetDetector
from training.detectors.xception_detector import XceptionDetector  # if needed later 
from training.detectors.vgg_detector import VGGDetector  # if needed later

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

class GradCamEvaluator:
    def __init__(self, model_name="resnet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name


        if model_name == "resnet":
            self.target_layers = [self.model.backbone.resnet[-1][-1].conv2]
        elif model_name == "xception":
            self.target_layers = [self.model.backbone.conv4.pointwise]
        elif model_name == "vgg":
            self.target_layers = [self.model.backbone.features[-1]]
        else:
            raise ValueError("Unsupported model name for Grad-CAM")

        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def extract_fake_position(self, path):
        try:
            return int(os.path.basename(path).split("_fake_")[1].split(".")[0])
        except Exception as e:
            logger.warning("Could not extract fake position from %s: %s", path, e)
            return -1

    def convert_to_numpy(self, tensor):
        tensor = tensor.squeeze(0)
        return (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    def evaluate_heatmap(self, tensor, path, grid_split, true_fake_pos):
        grayscale_cam = self.cam(input_tensor=tensor.unsqueeze(0),
                                 targets=[ClassifierOutputTarget(1)])[0]

        rgb_img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        intensity_sum = np.sum(grayscale_cam, axis=(0, 1))
        guessed_fake_pos = int(np.argmax(intensity_sum)) % grid_split

        return heatmap, guessed_fake_pos, intensity_sum

    def evaluate(self, tensor_list, path_list, grid_split):
        results = []
        logger.info("Processing %d grids with grid_split=%d", len(tensor_list), grid_split)

        for idx, (tensor, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)

            if tensor.shape[1] == 3:
                tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1)

            true_fake_pos = self.extract_fake_position(path)
            heatmap, guessed_fake_pos, intensity_sums = self.evaluate_heatmap(
                tensor[0], path, grid_split, true_fake_pos
            )

            original_image = self.convert_to_numpy(tensor[0])
            model_prediction = 1
            acc = 1 if guessed_fake_pos == true_fake_pos else 0

            result = {
                "path": path,
                "original_image": original_image,
                "heatmap": heatmap,
                "guessed_fake_position": guessed_fake_pos,
                "true_fake_position": true_fake_pos,
                "accuracy": acc,
                "model_prediction": model_prediction
            }

            logger.info("For grid %s: true pos %d, predicted %d, accuracy: %.3f",
                        os.path.basename(path), true_fake_pos, guessed_fake_pos, acc)

            results.append(result)

        return results
