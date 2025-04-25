import os
import sys

#set project path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

import logging
import torch
import argparse
import numpy as np
import pickle
import random
from PIL import Image
from Utils_PointingGame import load_model, load_config, preprocess_image, Analyser
from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  
from GradCam_eval import GradCamEvaluator
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

#######################
# set model path, config path and additional arguments
CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_MPG_config.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception_bcos.yaml")
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_minimal.yaml")
ADDITIONAL_ARGS = {
    "test_batchSize": 12
}
#######################

#setpup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MaskPointingGameCreator(Analyser):
    def __init__(self, base_output_dir, xai_method=None, plotting_only=False,
                 model=None, model_name="default", config_name="default",
                 test_data_loaders=None, dataset=None, device=None, config=None, overwrite=False, quantitativ=False, threshold_steps=0):
        """
        Initialize grid creator with specified parameters.
        base_output_dir: Base directory for grids.
        xai_method: a valid xai method
        plotting_only: If True, load existing results.
        """
        self.xai_method = xai_method
        self.model = model
        self.test_data_loaders = test_data_loaders
        self.dataset = dataset
        self.model_name = model_name
        self.config_name = config_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{config_name}")
        self.device = device
        self.overwrite = overwrite
        self.quantitativ = quantitativ
        self.threshold_steps = threshold_steps

        if plotting_only:
            self.load_results()
            return
        
        # Create output directory for grids.
        self.output_dir = os.path.join(self.output_folder, "MaskPointingGame")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load or compute sorted image rankings.
        self.ranking_file = os.path.join(self.output_folder, "sorted_confs.pkl")
        if os.path.exists(self.ranking_file):
            self.sorted_confs = self.load_ranking(self.ranking_file)
            logger.info("Loaded sorted confidences from %s", self.ranking_file)
        else:
            self.sorted_confs = self.compute_sorted_confs_fakes()
            self.save_ranking(self.sorted_confs, self.ranking_file)
            logger.info("Saved sorted confidences to %s", self.ranking_file)

    def analysis(self):
        """Analysis takes all images from data loader and plays the mask pointing game.
        It returns the a list of result dictionaries."""
        key = list(self.test_data_loaders.keys())[0]
        for data_dict in self.test_data_loaders[key]:
            # Move all tensor values in data_dict to the device first.
            for k, value in data_dict.items():
                if value is not None and hasattr(value, 'to'):
                    data_dict[k] = value.to(self.device)

            # Now unpack after moving to device.
            img_batch, label_batch, mask, landmark, path_of_image = (
                data_dict[k] for k in ['image', 'label', 'mask', 'landmark', 'image_path']
            )
            
            # Filter to get only fake images
            label_mask = (label_batch == 1)
            img_batch = img_batch[label_mask]
            label_batch = label_batch[label_mask]
            if mask is not None:
                mask = mask[label_mask]
            landmark = landmark[label_mask]
            
            # Remove extra key if present.
            data_dict.pop('label_spe', None)
            # Convert labels to binary.
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)

            num_samples = img_batch.shape[0]
            
            #initialize results list to be filled with result dict
            results = []
            
            # Process each image in the batch.
            for j in range(num_samples):
                image = img_batch[j].unsqueeze(0)  # shape: [1, C, H, W]
                label = label_batch[j]
                true_label = int(label.item())  # Convert label tensor to int.
                image_path = path_of_image[j]
                
                #load in mask bc sometimes none with dataloader
                mask =load_sample_by_path(image_path)[1]
                
                #heatmaps for all requested methods in dict of form {method: heatmap}
                heatmap = generate_heatmaps_for_methods(xai_method = self.xai_method, image=image)

                #Model class and model confidence
                logit = output['cls']  # Expected shape: [1, num_classes]
                # Get predicted label from the first (and only) sample.
                predicted_label = logit[0].argmax().item()
                # Compute confidence from the corresponding logit.
                confidence = logit[0, predicted_label].item()
                
                #Play MPG w/ thresholds
                thresholds = [None]  # No threshold
                if threshold_steps > 0:
                    thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]
                for t in thresholds:
                    logger.info("Evaluating with threshold: %s", t if t is not None else "no threshold")
                    acc, intensity_acc = mask_game(heatmap = heatmap, mask = mask)
                    result = {
                        "threshold": t if t is not None else 0,
                        "path": image_path,
                        "original_image": image,
                        "heatmap": heatmap,
                        "accuracy": acc,
                        "intensity_accuracy": intensity_acc,
                        "model_prediction": predicted_label,
                        "model_confidence": confidence,
                        "xai_method": xai_method
                    }
                    results.append(result)
        #calculate overall - Does it make sense even??
        grid_accuracies = [res["accuracy"] for res in raw_results]
        percentiles = np.percentile(np.array(grid_accuracies), [25, 50, 75, 100])
        logger.info("Localisation accuracy percentiles: %s", percentiles)
        overall = {"localisation_metric": grid_accuracies, "percentiles": percentiles}
        return overall, results
        
    def save_results(results):
        # f) group & pickle raw results by threshold
        threshold_groups = collections.defaultdict(list)
        for entry in results:
            thr = entry.get("threshold", None)
            threshold_groups[thr].append(entry)
    
        out_dir = os.path.join(OUTPUT_BASE_DIR, cfg["name"])
        os.makedirs(out_dir, exist_ok=True)
    
        all_raw_path = os.path.join(out_dir, "results_by_threshold.pkl")
        with open(all_raw_path, "wb") as f:
            pickle.dump(dict(threshold_groups), f)
        print(f" → saved grouped results: {all_raw_path}")

    def generate_heatmap_for_method(xai_method, image):
        """
        Generate a heatmap for each XAI method and return them in a dictionary.
    
        Args:
            xai_methods (list): List of strings (e.g., ["gradcam", "bcos"])
            image (torch.Tensor): The input image tensor [1, C, H, W]
    
        Returns:
            heatmap (tensor)
        """
        if method == "bcos":
            image = preprocess_image(image)
            evaluator = BCOSEvaluator(self.model, self.device)
        elif method == "lime":
            image = image[:,:3]
            evaluator = LIMEEvaluator(self.model, self.device)
        elif method == "gradcam":
            image = image[:,:3]
            raise NotImplementedError("GradCAM evaluator not implemented.") #fix this
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")      
        try:
            # Call your heatmap generator (you may need to adjust this call signature)
            heatmap = evaluator.generate_heatmap(image=image)
            logger.debug("Generated heatmap for method: %s | shape: %s", method, heatmap.shape)

        except Exception as e:
            logger.error("Error generating heatmap for method %s: %s", method, e)
        return heatmap
        
    def mask_game(mask, heatmap, threshold):
        """
        play the mask game for a given heatmap and threshold for both intensity-based and non-intensity based
        return the respective accuracies
        """
        if threshold == None:
            threshold = 0
        #without intensity
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()  # Convert tensor to numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  # Convert tensor to numpy array
    
        heatmap = heatmap[:, :, -1:].copy()
        intensity_map = heatmap[:,:,-1:].copy()
        # Ensure both are in the 0-1 range (binary)
        if np.max(heatmap) > 1:  # If values are in 0-255 (image format), threshold to 0 or 1
            print("heatmap range may be wrong")
            heatmap = np.where(heatmap > threshold, 1, 0)
    
        if np.max(mask) > 1:  # If values are in 0-255 (image format), threshold to 0 or 1
            print("mask range may be wrong")
            mask = np.where(mask > 0, 1, 0)
            
        # assuming mask is a tensor or numpy array
        correct_pixels = np.sum((heatmap == 1) & (mask == 1))  # Pixels that are both predicted as "1" and ground truth "1"
        # print(np.array((heatmap == 1) & (mask == 1)).shape)
        total_predicted_pixels = np.sum(heatmap == 1)  # Total ground truth pixels that are part of the mask
        # print(np.array(heatmap == 1).shape)
        # Step 5: Compute the performance metric (e.g., accuracy)
        accuracy = correct_pixels / total_predicted_pixels if total_predicted_pixels > 0 else 0  # Accuracy based on mask region
        
        # Optionally, calculate Intersection over Union (IoU) for better performance measurement
        intersection = correct_pixels
        union = np.sum((heatmap == 1) | (mask == 1))  # Union of predicted mask and ground truth mask
        iou = intersection / union if union > 0 else 0  # IoU

        #with intensity
        total_intensity = np.sum(intensity_map[intensity_map>threshold])
        mask_intensity = np.sum(intensity_map[mask ==1])
        non_mask_intensity = np.sum(intensity_map[mask==0])
        intensity_accuracy = mask_intensity/total_intensity if total_intensity > 0 else 0

        return accuracy.round(4), intensity_accuracy.round(4)

        

    def load_sample_by_path(self, image_path, expected_label):
        """
        Retrieve the sample (image, label, etc.) from the dataset by matching the stored image path.
        If the provided image_path is a single-element list, extract the string.
        """
        # If image_path is a list of one element, get the string.
        if isinstance(image_path, list) and len(image_path) == 1:
            image_path = image_path[0]
        
        try:
            idx = self.dataset.image_list.index(image_path)
        except ValueError:
            raise ValueError(f"Image path {image_path} not found in dataset.")
        
        # Retrieve the sample from the dataset using its __getitem__.
        sample = self.dataset[idx]  # Expected to be a tuple: (image, label, landmark, mask, stored_index)
        sample_label = int(sample[1])
        if sample_label != expected_label:
            raise ValueError(f"Label mismatch at {image_path}: expected {expected_label} but got {sample_label}")

        image = sample[0]
        mask = sample[3]
        return image, mask
    
def main():
    config = load_config(MODEL_PATH, CONFIG_PATH, additional_args=ADDITIONAL_ARGS)

    required_keys = ["overwrite", "quantitativ", "xai_method", "max_images"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    logger.info("Parameters: XAI=%s, Base=%s, Model=%s", config['xai_method'], config['base_output_dir'], MODEL_PATH)

    model = load_model(config)

    pretrained_path = config['pretrained']
    state_dict = torch.load(pretrained_path)
    # Remove "module." prefix from state_dict keys if necessary.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    ##########
    
    # 2) Lade die Gewichte **einmal** und fange das Ergebnis ab
    res = model.load_state_dict(new_state_dict, strict=False)
    
    # 3) Logge, was fehlt und was extra war
    logger.info("Missing keys: %s", res.missing_keys)
    logger.info("Unexpected keys: %s", res.unexpected_keys)
    
    # 4) quick‐check eines Backbone‐Weights
    first_weight = next(model.backbone.parameters())
    logger.info("Mean of first backbone weight tensor: %.6f", first_weight.mean().item())

    all_b = [m.b for m in model.backbone.modules() if hasattr(m, "b")]
    print(f"Gefundene b-Werte im ResNet: {set(all_b)}")  





    ########
    
    # Set device and move model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set model to evaluation mode.
    logger.info("Loaded model %s on device %s", model.__class__.__name__, device)

    ##########




    
    # 1. Zähle alle detach-fähigen Module
    all_detachable = [m for m in model.backbone.modules() if hasattr(m, "detach")]
    print("Anzahl aller detach-fähigen Module:", len(all_detachable))

    # 2. Vor Explanation-Modus: sollten alle detach=False sein
    pre = sum(1 for m in all_detachable if m.detach)
    print("Vorher detach=True:", pre)

    # 3. Im Explanation-Modus
    with model.backbone.explanation_mode():
        mid = sum(1 for m in all_detachable if m.detach)
        print("Im Kontext detach=True:", mid)

    # 4. Nach Exit: wieder alle auf False?
    post = sum(1 for m in all_detachable if m.detach)
    print("Danach detach=True:", post)
    # —––––––––– ENDE ––––––––––

    ############
        
    model_name = config.get("model_name", "defaultModel")
    config_name = os.path.basename(CONFIG_PATH).split('.')[0]

    # Prepare testing data.
    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = list(test_data_loaders.values())[0]
    dataset = test_loader.dataset

    MPG_creator.run() # Run analysis.
    MPG_creator.save_results()

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/MPG_eval.py