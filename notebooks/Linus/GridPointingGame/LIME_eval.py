import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import yaml
from lime import lime_image
from skimage.segmentation import mark_boundaries  # Explicit import


# Append the project path (adjust as needed)
# sys.path.append("/Users/toby/Interpretable-Deep-Fake-Detection")
sys.path.append("/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection")
sys.argv = ["train.py"]

# Import detectors configuration and models
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR


def load_config(path, additional_args={}):
    """
    Loads the YAML configuration file and updates it with additional arguments.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    try:
        with open('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
            config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    config.update(config2)
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
    for key, value in additional_args.items():
        config[key] = value
    return config

class LIMEEvaluator:
    """
    LIMEEvaluator loads a detector model using the provided configuration
    and provides methods for generating LIME explanations on pre-saved grid tensors.
    This module is structured similarly to your BCos evaluator so that it can be 
    seamlessly integrated in your GPG_eval pipeline when xai_method=="lime".
    """
    def __init__(self, config_path, additional_args, xai_method="lime"):
        # Load configuration and initialize the model.
        self.config = load_config(config_path, additional_args=additional_args)
        print("Registered models:", DETECTOR.data.keys())
        model_class = DETECTOR[self.config['model_name']]
        self.model = model_class(self.config)
        self.model.eval()
        self.xai_method = xai_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[DEBUG] Loaded {self.model.__class__.__name__} model onto {self.device}")

        # Define a transform for LIME processing (resize to 224x224 and convert to tensor)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        # Create a LIME explainer instance.
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        """
        Prediction function for LIME.
        Args:
            images: List or numpy array of images in HWC format (values 0-255).
        Returns:
            Numpy array of prediction probabilities with shape [N, 2].
        """
        processed_images = []
        for img in images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            tensor_img = self.transform(pil_img)  # Output: [C, 224, 224]
            processed_images.append(tensor_img.numpy())
        batch = np.stack(processed_images, axis=0)  # [N, C, 224, 224]
        batch = torch.from_numpy(batch).to(self.device)
        data = {'image': batch}
        with torch.no_grad():
            output = self.model(data)
            probs = output['prob'].cpu().numpy()  # [N]
            # Convert to two-column probabilities: [negative, positive]
            probs = np.vstack([1 - probs, probs]).T
        return probs

    def lime_grid_eval(self, heatmap, background_pixel=0):
        """
        Evaluates a grid image by splitting it into 4 sections.
        Returns:
            fake_pred_index: Index of the section with the highest count of non-background pixels.
            non_0_pixel_count: List of counts per section.
        """
        heatmap_gray = np.mean(heatmap, axis=2)
        if heatmap_gray.max() <= 1.0:
            heatmap_gray = (heatmap_gray * 255).astype(np.uint8)
        rows, cols = heatmap_gray.shape
        if rows != cols:
            raise ValueError("The heatmap must be square.")
        if rows % 2 != 0:
            raise ValueError("The heatmap dimensions must be divisible by 2.")
        half = rows // 2
        sections = [
            heatmap_gray[:half, :half],
            heatmap_gray[:half, half:],
            heatmap_gray[half:, :half],
            heatmap_gray[half:, half:]
        ]
        non_0_pixel_count = [np.sum(section > background_pixel) for section in sections]
        fake_pred_index = np.argmax(non_0_pixel_count)
        return fake_pred_index, non_0_pixel_count

    def evaluate(self, tensor_list, path_list, grid_split):
        """
        Main evaluation loop.
        Iterates over pre-loaded grid tensors (from .pt files) and their paths,
        generates LIME explanations, and displays side-by-side comparisons.
        
        Args:
            tensor_list (list): List of pre-loaded grid tensors.
            path_list (list): List of corresponding file paths.
            grid_split (int): Grid split parameter for evaluation.
        """
        for tensor, path in zip(tensor_list, path_list):
            print(f"[DEBUG] Processing file: {path}")
            # Expect tensor shape [1, 3, H, W] (loaded from .pt file)
            img = tensor.to(self.device)  # Use the tensor as-is.
            print(f"[DEBUG] Image tensor shape: {img.shape}")
            # Convert tensor to numpy image in HWC format.
            img_np = np.transpose(img[0, ...].cpu().numpy(), (1, 2, 0))
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img.requires_grad_(True)

            data_dict = {'image': img, 'label': 0}  # Dummy label
            self.model.zero_grad()
            out = self.model(data_dict)

            # Generate LIME explanation for the image.
            explanation = self.explainer.explain_instance(
                img_np,
                self.batch_predict,
                top_labels=1,
                hide_color=0,
                num_samples=10
            )
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=10,
                hide_rest=True
            )
            fake_pred, pixel_counts = self.lime_grid_eval(temp)
            print(f"[DEBUG] LIME grid evaluation for {path}: fake_pred: {fake_pred}, pixel counts: {pixel_counts}")
            print(f"[DEBUG] Top predicted label: {top_label}")

            # Display side-by-side: original image and LIME explanation.
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            try:
                true_fake_pos = int(os.path.basename(path).split('_fake_')[1].split('.')[0])
            except Exception as e:
                true_fake_pos = -1
            ax[0].imshow(img_np)
            ax[0].axis('off')
            ax[0].set_title(f"Original Image: {true_fake_pos}")
            ax[1].imshow(mark_boundaries(temp, mask))
            ax[1].axis('off')
            ax[1].set_title(f'LIME Explanation Prediction: {fake_pred}')
            plt.show()