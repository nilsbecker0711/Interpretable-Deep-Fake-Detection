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
from GradCam_evalnew import GradCamEvaluator
#from GradCam_evalbcos import GradCamEvaluator
#from GradCam_evallayer import GradCamEvaluator
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import collections

#######################
# set model path, config path and additional arguments
CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_MPG_bcos_1_75.yaml")

#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception_bcos.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_minimal.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_2_5_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_1_25_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_2_best_hpo.yaml")
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_1_75_best_hpo.yaml") #change with_mask: true, dataset_type: 'bcos', pretrained_weights path set
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet_bcos_minimal.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/vit_bcos_1_25_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/vit_bcos_1_75_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/vit_bcos_2_5_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/vit_bcos_2_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/vit_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception_bcos.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception.yaml")


#CHANGE MODEL YAML TO WITH_MASK = TRUE and DATASET_TYPE = 'bcos'

ADDITIONAL_ARGS = {
    "test_batchSize": 12
}
#######################

#setpup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaskPointingGameCreator(Analyser):
    def __init__(self, base_output_dir, xai_method=None, plotting_only=False,
                 model=None, model_name="default", config_name="default",
                 test_data_loaders=None, dataset=None, device=None, config=None, overwrite=False, quantitativ=False, threshold_steps=0, max_images = None, mask_resolution=224):
        """
        Initialize grid creator with specified parameters.
        base_output_dir: Base directory for grids.
        xai_method: a valid xai method
        plotting_only: If True, load existing results.
        """
        self.xai_method = xai_method.lower().strip()
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
        self.max_images = max_images
        self.results_dir = os.path.join(self.output_folder, f"MaskPointingGame")
        self.mask_resolution = mask_resolution

        if plotting_only:
            self.load_results()
            return
        
        # Create output directory for grids.
        self.output_dir = os.path.join(self.output_folder, "MaskPointingGame")
        os.makedirs(self.output_dir, exist_ok=True)

    def analysis(self):
        """Analysis takes all images from data loader and plays the mask pointing game.
        It returns the a list of result dictionaries."""
        key = list(self.test_data_loaders.keys())[0]
        processed_images = 0
        results = []
        for data_dict in self.test_data_loaders[key]:
            # Move all tensor values in data_dict to the device first.
            for k, value in data_dict.items():
                if value is not None and hasattr(value, 'to'):
                    data_dict[k] = value.to(self.device)

            # Now unpack after moving to device.
            img_batch, label_batch, mask, landmark, path_of_image = (
                data_dict[k] for k in ['image', 'label', 'mask', 'landmark', 'image_path']
            )
            logger.debug(f"raw mask: {mask}")
            # Remove extra key if present.
            data_dict.pop('label_spe', None)
            # Convert labels to binary.
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)

            num_samples = img_batch.shape[0]

            # Process each image in the batch.
            for j in range(num_samples):
                label = label_batch[j]
                #only process fake labels for MPG
                if label == 0:
                    continue
                image = img_batch[j].unsqueeze(0)  # shape: [1, C, H, W]
                logger.debug("Sample %d | Label: %s", j, label.item())
                true_label = int(label.item())  # Convert label tensor to int.
                image_path = path_of_image[j]

                try:
                    #mask = self.load_sample_by_path(image_path, expected_label=1)[1]
                    mask = mask[j]
                    logger.debug(f"raw mask: {mask}")
                    mask = mask.squeeze()
                    if mask.shape != torch.Size([self.mask_resolution, self.mask_resolution]):
                        raise ValueError(f"Mask shape is {mask.shape}, expected torch.Size([{self.mask_resolution},{self.mask_resolution})")
                    if torch.max(mask) == 0:
                        raise ValueError(f"Mask has max value of {torch.max(mask)}")
                    logger.info(f"Mask loaded with shape: {mask.shape}")
                except Exception as e:
                    print(f"Error loading or processing mask for image {image_path}: {e}")
                     #print(f"Rejected Mask type: {type(mask)}, mask: {mask}")
                    continue
                
                #logger.debug(f"mask: {mask}")
                logger.debug(f"image path: {image_path}")
                original_image = image.clone()
                original_image = original_image[:,:3].squeeze()
                #preprocess image and then generate heatmap
                if self.xai_method == "bcos":
                    image = preprocess_image(image)
                elif self.xai_method in ["lime", "gradcam", "xgrad", "grad++", "layergrad"]:
                    image = image[:,:3]
                    #CHANGE FOR BCOS GRADCAM RUNS:
                    #image = preprocess_image(image)
                else:
                    raise ValueError(f"Unknown xai_method: {self.xai_method}")   
                heatmap = self.generate_heatmap_for_method(self.xai_method,image)

                #Model class and model confidence
                output = self.model({'image': image, 'label': label})
                logit = output['cls']  # Expected shape: [1, num_classes]
                # Get predicted label from the first (and only) sample.
                predicted_label = logit[0].argmax().item()
                # Compute confidence from the corresponding logit.
                confidence = logit[0, predicted_label].item()
                #Play MPG w/ thresholds
                thresholds = [None]  # No threshold
                if self.threshold_steps > 0:
                    thresholds += [i / self.threshold_steps for i in range(1, self.threshold_steps + 1)]
                for t in thresholds:
                    threshold_value = t if t is not None else 0
                    logger.info("Evaluating with threshold: %s", t if t is not None else "no threshold")
                    #apply threshold to map and zero out values below
                    thresholded_map = heatmap.copy()
                    thresholded_map[thresholded_map < threshold_value] = 0
                    acc, intensity_acc = self.mask_game(mask, thresholded_map)
                    logger.info(f"Unweighted accuracy: {acc}")
                    logger.info(f"Weighted Accuracies: {intensity_acc}")
                    result = {
                        "threshold": threshold_value,
                        "path": image_path,
                        #"original_image": original_image,
                        #"heatmap": thresholded_map,
                        "unweighted_localization_score": acc,
                        "weighted_localization_score": intensity_acc,
                        "model_prediction": predicted_label,
                        "model_confidence": confidence,
                        "xai_method": self.xai_method,
                        #"mask" : mask
                    }
                    results.append(result)
                    processed_images += 1
                    logger.info(f"{processed_images} have been processed so far!")
                if self.max_images is not None and processed_images >= self.max_images:
                    logger.info(f"Reached max_images={self.max_images}, exiting early.")
                    return results 
                        
        return results 
        

    def generate_heatmap_for_method(self, xai_method, image):
        """
        Generate a heatmap for each XAI method and return them in a dictionary.
    
        Args:
            xai_methods (string): A string representing a valid XAI method
            image (torch.Tensor): The input image tensor [1, C, H, W]
    
        Returns:
            heatmap (tensor)
        """  
        if xai_method == "bcos":
            evaluator = BCOSEvaluator(self.model, self.device)
        elif xai_method == "lime":
            evaluator = LIMEEvaluator(self.model, self.device) #how do i get intensity map here for LIME
        elif xai_method in ["gradcam", "xgrad", "grad++", "layergrad"]:
            image = image.squeeze(0)
            evaluator = GradCamEvaluator(self.model, self.device)
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")   
        try:
            # Call your heatmap generator (you may need to adjust this call signature)
            heatmap = evaluator.generate_heatmap(image)[0]
            logger.debug("Generated heatmap for method: %s | shape: %s, type: %s", xai_method, heatmap.shape, type(heatmap))

        except Exception as e:
            logger.error("Error generating heatmap for method %s: %s", xai_method, e)
        return heatmap
        
    def mask_game(self, mask, heatmap):
        """
        play the mask game for a given heatmap for both intensity-based and non-intensity based
        return the respective accuracies
        """
        #without intensity
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()  # Convert tensor to numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  # Convert tensor to numpy array
        if self.xai_method == 'bcos':
            intensity_map = heatmap[:,:,-1].copy()
            heatmap = heatmap[:, :, -1].copy()
        else:
            intensity_map = heatmap.copy()
        #logger.debug(f"Intensity map shape: {intensity_map.shape}")
        # Ensure both are in the 0-1 range (binary)
        # if np.max(heatmap) > 1:  # If values are in 0-255 (image format), threshold to 0 or 1
        #     print("heatmap range may be wrong")
        #     heatmap = np.where(heatmap > 0, 1, 0)
    
        # if np.max(mask) > 1:  # If values are in 0-255 (image format), threshold to 0 or 1
        #     print("mask range may be wrong")
        #     mask = np.where(mask > 0, 1, 0)
            
        #Without Intensity
        heatmap = (heatmap > 0).astype(np.uint8)
        correct_pixels = np.sum((heatmap == 1) & (mask == 1))  
        total_predicted_pixels = np.sum(heatmap == 1) 
        accuracy = correct_pixels / total_predicted_pixels if total_predicted_pixels > 0 else 0  # Accuracy based on mask region
        logger.debug(f"mask game accuracy for unweighted: {accuracy}")
        
        # Optionally, calculate Intersection over Union (IoU) for better performance measurement
        intersection = correct_pixels
        union = np.sum((heatmap == 1) | (mask == 1))  # Union of predicted mask and ground truth mask
        iou = intersection / union if union > 0 else 0  # IoU

        #with intensity
        total_intensity = np.sum(intensity_map)
        mask_intensity = np.sum(intensity_map[mask ==1])
        non_mask_intensity = np.sum(intensity_map[mask==0])
        intensity_accuracy = mask_intensity/total_intensity if total_intensity > 0 else 0
        logger.info(f"total intensity: {total_intensity}, mask_intensity: {mask_intensity}, non_mask_intensity: {non_mask_intensity}")
        logger.debug(f"mask max: {np.max(mask)}")
        try:
            accuracy = round(accuracy, 4)
        except TypeError:
            pass  # if it's an int, do nothing
        
        try:
            intensity_accuracy = round(intensity_accuracy, 4)
        except TypeError:
            pass
        
        return accuracy, intensity_accuracy
            
            
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

    required_keys = ["overwrite", "quantitativ", "xai_method"]
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
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Set device and move model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set model to evaluation mode.
    logger.info("Loaded model %s on device %s", model.__class__.__name__, device)
        
    model_name = config.get("model_name", "defaultModel")
    config_name = os.path.basename(CONFIG_PATH).split('.')[0]

    # Prepare testing data.
    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = list(test_data_loaders.values())[0]
    dataset = test_loader.dataset

    MPG_creator = MaskPointingGameCreator(
        base_output_dir=config.get("base_output_dir", "results"),
        xai_method=config["xai_method"],
        model=model,
        model_name=model_name,
        config_name=config_name,
        test_data_loaders=test_data_loaders,
        dataset=dataset,
        device=device,
        overwrite=config["overwrite"],
        quantitativ=config["quantitativ"],
        threshold_steps= config["threshold_steps"],
        max_images = config["max_images"],
        mask_resolution = config["mask_resolution"]
    )
    
    MPG_creator.run() # Run analysis.

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/MPG_eval.py
