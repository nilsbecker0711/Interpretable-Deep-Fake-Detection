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

sys.path.append("/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection")

from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR

class LIMEEvaluator:
    """
    LIMEEvaluator loads a detector model using the provided configuration
    and provides methods for generating LIME explanations on pre-saved grid tensors.
    """
    def __init__(self, config, model=None):
        self.config = config
        if model is None:
            print("Registered models:", DETECTOR.data.keys())
            model_class = DETECTOR[self.config['model_name']]
            self.model = model_class(self.config)
        else:
            self.model = model
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[DEBUG] Loaded {self.model.__class__.__name__} model onto {self.device}")

        self.transform = transforms.ToTensor()
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
        for idx, img in enumerate(images):
            pil_img = Image.fromarray(img.astype(np.uint8))
            tensor_img = self.transform(pil_img)
            processed_images.append(tensor_img.numpy())
        batch = np.stack(processed_images, axis=0)
        batch = torch.from_numpy(batch).to(self.device)
        data = {'image': batch}
        with torch.no_grad():
            output = self.model(data)
            probs = output['prob'].cpu().numpy()  # [N]
            # Convert to two-column probabilities: [negative, positive]
            probs = np.vstack([1 - probs, probs]).T
        return probs

    def lime_grid_eval(self, heatmap, grid_split=3, background_pixel=0):
        """
        Evaluates a grid image by splitting it into Gridsize sections.
        Returns:
            fake_pred_index: Index of the section with the highest count of non-background pixels.
            non_0_pixel_count: List of counts per section.
        """
        heatmap_gray = np.mean(heatmap, axis=2)
        if heatmap_gray.max() <= 1.0:
            heatmap_gray = (heatmap_gray * 255).astype(np.uint8)

        # Upscale if needed to ensure even grid divisions.
        original_size = heatmap_gray.shape  # (H, W)
        print(f"[DEBUG][lime_grid_eval] Original heatmap size: {original_size}")
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
            import cv2
            heatmap_gray = cv2.resize(heatmap_gray, new_size, interpolation=cv2.INTER_LINEAR)
            print(f"[DEBUG][lime_grid_eval] Resized heatmap_gray to: {new_size}")
        else:
            print(f"[DEBUG][lime_grid_eval] No resizing needed.")
        
        rows, cols = heatmap_gray.shape
        section_size_row = rows // grid_split
        section_size_col = cols // grid_split
        print(f"[DEBUG][lime_grid_eval] Section size: {section_size_row} x {section_size_col}")
        sections = []
        for i in range(grid_split):
            for j in range(grid_split):
                section = heatmap_gray[i * section_size_row:(i + 1) * section_size_row,
                                         j * section_size_col:(j + 1) * section_size_col]
                sections.append(section)

        
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
            
            important_pixels = temp[mask == 1]

            fake_pred, pixel_counts = self.lime_grid_eval(temp, grid_split=grid_split)

            # Display side-by-side: original image and LIME explanation.
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            try:
                true_fake_pos = int(os.path.basename(path).split('_fake_')[1].split('.')[0])
            except Exception as e:
                true_fake_pos = -1
                print(f"[DEBUG] Could not parse true fake position from filename: {e}")

            # Calculate grid accuracy as the ratio of nonzero pixels in the true cell to the total nonzero pixels.
            total_nonzero = float(sum(pixel_counts))
            if total_nonzero > 0 and 0 <= true_fake_pos < len(pixel_counts):
                grid_accuracy = pixel_counts[true_fake_pos] / total_nonzero
            else:
                grid_accuracy = 0
            print(f"[DEBUG] For grid {os.path.basename(path)}: true position {true_fake_pos}, predicted {fake_pred}, grid metric: {grid_accuracy:.3f}")

            # Left: Original image
            ax[0].imshow(img_np)
            ax[0].axis('off')
            ax[1].set_title(f'LIME Prediction: {fake_pred}\nGrid metric: {grid_accuracy:.3f}')

            # Right: LIME explanation with grid lines
            # Convert temp to float in [0,1] for proper visualization.
            temp_norm = temp.astype(np.float32) / 255.0
            boundaries_img = mark_boundaries(temp_norm, mask)
            ax[1].imshow(boundaries_img)
            # Ensure the axis limits match the image dimensions.
            height, width, _ = temp_norm.shape
            ax[1].set_xlim(0, width)
            ax[1].set_ylim(height, 0)
            ax[1].set_aspect('equal')

            # Draw grid lines.
            for i in range(1, grid_split):
                y = i * height / grid_split
                x = i * width / grid_split
                ax[1].hlines(y, 0, width, colors='grey', linestyles='dashed', linewidth=0.5)
                ax[1].vlines(x, 0, width, colors='grey', linestyles='dashed', linewidth=0.5)

            ax[1].axis('off')
            ax[1].set_title(f'LIME Explanation Prediction: {fake_pred}')
            plt.tight_layout()
            plt.show()
