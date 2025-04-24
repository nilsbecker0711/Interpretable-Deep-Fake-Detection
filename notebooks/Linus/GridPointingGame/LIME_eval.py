import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from lime import lime_image
import logging
from skimage.segmentation import mark_boundaries

# Setup logging and project root
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class LIMEEvaluator:
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

        original_size = heatmap.shape  # (H, W)
        logger.debug("[LIME] Original heatmap size: %s", original_size)
        
        # Resize if dimensions aren't divisible by grid_split
        if original_size[1] % grid_split != 0 or original_size[0] % grid_split != 0:
            new_size = (original_size[1] * grid_split, original_size[0] * grid_split)
            import cv2
            heatmap = cv2.resize(heatmap, new_size, interpolation=cv2.INTER_LINEAR)
            logger.debug("[LIME] Resized heatmap to: %s", new_size)
        else:
            logger.debug("[LIME] No resizing needed.")
        
        # Split image into grid sections
        rows, cols = heatmap.shape
        section_size_row = rows // grid_split
        section_size_col = cols // grid_split
        sections = [
            heatmap[i * section_size_row:(i + 1) * section_size_row,
                         j * section_size_col:(j + 1) * section_size_col]
            for i in range(grid_split) for j in range(grid_split)
        ]
        print(np.max(heatmap))
        
        # Count non-background pixels per section and choose the section with the maximum count
        grid_intensity_sums = [np.sum(section) for section in sections]  # sum of intensities, not count
        for i, grid_intensity in enumerate(grid_intensity_sums):
            print("Intensitätssumme für Zelle {}: {}".format(i, grid_intensity))
            
        fake_pred_index = np.argmax(grid_intensity_sums)
        return fake_pred_index, grid_intensity_sums
    
    def extract_fake_position(self, path):
        """Extract fake position from filename."""
        try:
            return int(os.path.basename(path).split('_fake_')[1].split('.')[0])
        except Exception as e:
            logger.warning("Could not extract fake position from '%s': %s", path, e)
            return -1

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0):
        results = []
        for tensor, path in zip(tensor_list, path_list):
            logger.info("Processing file: %s", path)
            img = tensor.to(self.device)
            img_np = np.transpose(img[0, ...].cpu().numpy(), (1, 2, 0))
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img.requires_grad_(True)

            data_dict = {'image': img, 'label': 0}
            self.model.zero_grad()
            out = self.model(data_dict)
            model_prediction = out['cls'][0].argmax().item()

            
            explanation = self.explainer.explain_instance(
                img_np, self.batch_predict, top_labels=2, hide_color=0, num_samples=500
            )
            fake_label = 1
            weights = dict(explanation.local_exp[fake_label])
            
            weight_values = list(weights.values())
            print(f"Min weight: {min(weight_values):.6f}, Max weight: {max(weight_values):.6f}")
            
            segments = explanation.segments
            
            weights = dict(explanation.local_exp[fake_label])  # {segment_idx: weight}

            # Extrahiere positive Gewichte
            positive_weights = [w for w in weights.values() if w >= 0]
            
            # Bestimme das maximale positive Gewicht
            max_positive_weight = max(positive_weights) if positive_weights else 0
            
            # Skaliere die Gewichte relativ zum maximalen positiven Gewicht
            scaled_weights = {k: (v / max_positive_weight) if v > 0 else 0 for k, v in weights.items()}

            intensity_map = np.zeros(segments.shape, dtype=np.float32)
            for seg_val in np.unique(segments):
                if seg_val in scaled_weights:
                    intensity_map[segments == seg_val] = scaled_weights[seg_val]
    
            thresholds = [None]
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]
    
            for t in thresholds:
                threshold_value = t if t is not None else 0

                thresholded_map = intensity_map.copy()

                # Zero out values below the threshold
                thresholded_map[thresholded_map < threshold_value] = 0
                    
                true_fake_pos = self.extract_fake_position(path)
                fake_pred, grid_intensity_sums = self.lime_grid_eval(
                    thresholded_map, grid_split=grid_split, background_pixel=0.0
                )
                total_intensity = sum(grid_intensity_sums)
                if total_intensity > 0 and 0 <= true_fake_pos < len(grid_intensity_sums):
                    grid_accuracy = grid_intensity_sums[true_fake_pos] / total_intensity
                else:
                    grid_accuracy = 0

                result = {
                    "threshold": t,
                    "path": path,
                    "original_image": img_np,
                    "heatmap": thresholded_map,
                    "guessed_fake_position": fake_pred,
                    "accuracy": grid_accuracy,
                    "true_fake_position": true_fake_pos,
                    "model_prediction": model_prediction
                }
                results.append(result)

                logger.info("Threshold %s | %s: true pos %d, predicted %d, accuracy: %.3f",
                            str(t), os.path.basename(path), true_fake_pos, fake_pred, grid_accuracy)

        return results