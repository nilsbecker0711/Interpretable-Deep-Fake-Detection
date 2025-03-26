import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from lime import lime_image
import logging

# Setup logging and project root
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class LIMEEvaluator:
    """Evaluator that generates LIME explanations for grid tensors."""
    
    def __init__(self, model=None, device=None):
        """Initialize the evaluator: set up model, device, transform, and LIME explainer."""
        self.model = model
        self.device = device
        logger.info("Loaded model %s onto %s", self.model.__class__.__name__, self.device)
        self.transform = transforms.ToTensor()
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        """
        Prediction function for LIME.
        Converts perturbed images (HWC, 0-255) to tensors, runs the model, and returns prediction probabilities.
        """
        processed_images = [
            self.transform(Image.fromarray(img.astype(np.uint8))).numpy()
            for img in images
        ]
        batch = np.stack(processed_images, axis=0)
        batch = torch.from_numpy(batch).to(self.device)
        data = {'image': batch}
        with torch.no_grad():
            output = self.model(data)
            probs = output['prob'].cpu().numpy()  # [N]
            probs = np.vstack([1 - probs, probs]).T  # Create two-column probability array
        return probs

    def lime_grid_eval(self, heatmap, grid_split=3, background_pixel=0):
        """
        Evaluates the heatmap by splitting it into a grid and counting non-background pixels in each cell.
        Returns the index of the grid cell with the highest count and the counts per section.
        """
        # Convert heatmap to grayscale
        heatmap_gray = np.mean(heatmap, axis=2)
        if heatmap_gray.max() <= 1.0:
            heatmap_gray = (heatmap_gray * 255).astype(np.uint8)

        original_size = heatmap_gray.shape  # (H, W)
        logger.debug("[LIME] Original heatmap size: %s", original_size)
        
        # Resize if dimensions aren't divisible by grid_split
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
            import cv2
            heatmap_gray = cv2.resize(heatmap_gray, new_size, interpolation=cv2.INTER_LINEAR)
            logger.debug("[LIME] Resized heatmap_gray to: %s", new_size)
        else:
            logger.debug("[LIME] No resizing needed.")
        
        # Split image into grid sections
        rows, cols = heatmap_gray.shape
        section_size_row = rows // grid_split
        section_size_col = cols // grid_split
        sections = [
            heatmap_gray[i * section_size_row:(i + 1) * section_size_row,
                         j * section_size_col:(j + 1) * section_size_col]
            for i in range(grid_split) for j in range(grid_split)
        ]
        
        # Count non-background pixels per section and choose the section with the maximum count
        non_0_pixel_count = [np.sum(section > background_pixel) for section in sections]
        fake_pred_index = np.argmax(non_0_pixel_count)
        return fake_pred_index, non_0_pixel_count

    def evaluate(self, tensor_list, path_list, grid_split):
        """
        Evaluates a list of grid tensors with LIME.
        For each tensor, generates a LIME explanation and computes an accuracy metric based on grid sections.
        """
        results = []
        for tensor, path in zip(tensor_list, path_list):
            logger.info("Processing file: %s", path)
            img = tensor.to(self.device)  # Send tensor to device
            logger.debug("Image tensor shape: %s", img.shape)
            # Convert tensor to numpy array in HWC format
            img_np = np.transpose(img[0, ...].cpu().numpy(), (1, 2, 0))
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img.requires_grad_(True)

            # Dummy data for label and forward pass to zero gradients
            data_dict = {'image': img, 'label': 0}
            self.model.zero_grad()
            out = self.model(data_dict)

            # Generate LIME explanation for the image
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
            
            # Evaluate the LIME heatmap using grid splitting
            fake_pred, pixel_counts = self.lime_grid_eval(temp, grid_split=grid_split)

            # Parse true fake position from the file name
            try:
                true_fake_pos = int(os.path.basename(path).split('_fake_')[1].split('.')[0])
            except Exception as e:
                logger.warning("Could not parse true fake position from filename %s: %s", path, e)
                true_fake_pos = -1

            total_nonzero = float(sum(pixel_counts))
            if total_nonzero > 0 and 0 <= true_fake_pos < len(pixel_counts):
                grid_accuracy = pixel_counts[true_fake_pos] / total_nonzero
            else:
                grid_accuracy = 0
            logger.info("For grid %s: true position %d, predicted %d, grid accuracy: %.3f",
                        os.path.basename(path), true_fake_pos, fake_pred, grid_accuracy)

            # Store the results in a dictionary
            result = {
                "path": path,
                "original_image": img_np,  # Original image in HWC format
                "heatmap": temp,           # LIME explanation (heatmap)
                "guessed_fake_position": fake_pred,
                "accuracy": grid_accuracy,
                "true_fake_position": true_fake_pos
            }
            results.append(result)
            logger.info("Processed %s.", os.path.basename(path))

        return results