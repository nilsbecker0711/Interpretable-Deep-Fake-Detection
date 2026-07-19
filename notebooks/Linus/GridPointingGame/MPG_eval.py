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
import cv2
import pickle
import random
from PIL import Image
from Utils_PointingGame import load_model, load_config, preprocess_image, Analyser
from training.utils.xai.B_COS_eval import BCOSEvaluator
from training.utils.xai.LIME_eval import LIMEEvaluator
from training.utils.xai.GradCam_eval import GradCamEvaluator
from training.utils.xai.xai_common import mpg_mask_game
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import collections

#######################
# Model/config selection happens via CLI arguments (see parse_args below), e.g.:
#   python MPG_eval.py --model-config training/config/detector/resnet34_bcos_v2.yaml \
#                      --test-config results/test_MPG_bcos_1_75.yaml \
#                      --weights path/to/ckpt_best.pth --xai-method bcos
# The model yaml needs with_mask: true and dataset_type: 'bcos'; both can also
# be forced via --set (e.g. --set with_mask=true).
#######################

import yaml


def resolve_path(path):
    """Interpret relative paths as relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_PATH, path)


def parse_cli_overrides(pairs):
    """Turn repeated --set KEY=VALUE flags into a config-override dict."""
    overrides = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Mask Pointing Game evaluation")
    parser.add_argument("--model-config", required=True,
                        help="Detector yaml, e.g. training/config/detector/resnet34_bcos_v2.yaml")
    parser.add_argument("--test-config", required=True,
                        help="Run overlay yaml, e.g. results/test_MPG_bcos_1_75.yaml")
    parser.add_argument("--weights", default=None,
                        help="Checkpoint .pth to evaluate; overrides 'pretrained' from the yamls")
    parser.add_argument("--xai-method", default=None,
                        choices=["bcos", "gradcam", "xgrad", "grad++", "layergrad", "lime"],
                        help="Overrides xai_method from the test config")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides base_output_dir from the test config")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="test_batchSize override")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra config overrides (repeatable), e.g. "
                             "--set dataset_json_folder=preprocessing/dataset_json_v3")
    return parser.parse_args()

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
        self.config = config or {}
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
                    # IMPORTANT: do NOT overwrite the batch-level `mask` tensor here.
                    # Doing so made every later sample in the batch index into the
                    # previous sample's mask, fail the shape check, and get skipped —
                    # so only the first fake per batch was ever evaluated.
                    sample_mask = mask[j]
                    logger.debug(f"raw mask: {sample_mask}")
                    sample_mask = sample_mask.squeeze()
                    if sample_mask.shape != torch.Size([self.mask_resolution, self.mask_resolution]):
                        raise ValueError(f"Mask shape is {sample_mask.shape}, expected torch.Size([{self.mask_resolution},{self.mask_resolution}])")
                    if torch.max(sample_mask) == 0:
                        raise ValueError(f"Mask has max value of {torch.max(sample_mask)}")
                    logger.info(f"Mask loaded with shape: {sample_mask.shape}")
                except Exception as e:
                    logger.warning(f"Error loading or processing mask for image {image_path}: {e}")
                    continue
                
                #logger.debug(f"mask: {mask}")
                logger.debug(f"image path: {image_path}")
                original_image = image.clone()
                original_image = original_image[:,:3].squeeze()
                # The image comes from the model's OWN dataloader and is already
                # in the model's input space (bcos: 6ch [0,1]+inverse, standard:
                # 3ch normalized). The input must match the MODEL, not the XAI
                # method — the old per-method channel slicing broke CAM methods
                # on 6-channel b-cos models (and needed manual toggling).
                heatmap = self.generate_heatmap_for_method(self.xai_method, image)

                #Model class and model confidence
                output = self.model({'image': image, 'label': label})
                logit = output['cls']  # Expected shape: [1, num_classes]
                # Get predicted label from the first (and only) sample.
                predicted_label = logit[0].argmax().item()
                # Softmax confidence of the predicted class (consistent with GPG's ranking).
                confidence = torch.softmax(logit, dim=1)[0, predicted_label].item()
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
                    acc, intensity_acc = self.mask_game(sample_mask, thresholded_map)
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
                        #"mask" : sample_mask
                    }
                    results.append(result)
                # count once per image, not once per threshold; otherwise max_images
                # is exhausted after max_images/len(thresholds) actual images
                processed_images += 1
                logger.info(f"{processed_images} images have been processed so far!")
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
            evaluator = LIMEEvaluator(self.model, self.device,
                                      mean=self.config.get('mean', [0.5, 0.5, 0.5]),
                                      std=self.config.get('std', [0.5, 0.5, 0.5]))
        elif xai_method in ["gradcam", "xgrad", "grad++", "layergrad"]:
            image = image.squeeze(0)
            # pass the actual method so xgrad/grad++/layergrad are not silently run as gradcam
            evaluator = GradCamEvaluator(self.model, self.device, method=xai_method)
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")
        # Call the heatmap generator; index [0] is the 2D scored map for every method.
        heatmap = evaluator.generate_heatmap(image)[0]
        logger.debug("Generated heatmap for method: %s | shape: %s, type: %s", xai_method, heatmap.shape, type(heatmap))
        return heatmap
        
    def mask_game(self, mask, heatmap):
        """
        play the mask game for a given heatmap for both intensity-based and non-intensity based
        return the respective accuracies.
        Delegates to the shared single-source implementation in
        training.utils.xai.xai_common (also used by the in-training XAI monitor).
        """
        return mpg_mask_game(mask, heatmap)
            
            
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
    args = parse_args()
    model_path = resolve_path(args.model_config)
    config_path = resolve_path(args.test_config)

    additional_args = {"test_batchSize": args.batch_size}
    additional_args.update(parse_cli_overrides(args.set))
    if args.weights is not None:
        additional_args["pretrained"] = resolve_path(args.weights)
    if args.xai_method is not None:
        additional_args["xai_method"] = args.xai_method
    if args.output_dir is not None:
        additional_args["base_output_dir"] = resolve_path(args.output_dir)

    config = load_config(model_path, config_path, additional_args=additional_args)

    required_keys = ["overwrite", "quantitativ", "xai_method"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    logger.info("Parameters: XAI=%s, Base=%s, Model=%s", config['xai_method'], config['base_output_dir'], model_path)

    model = load_model(config)

    pretrained_path = config['pretrained']
    if not pretrained_path:
        raise ValueError("No checkpoint given: pass --weights or set 'pretrained' in a yaml")
    state_dict = torch.load(pretrained_path)
    # Remove "module." prefix from state_dict keys if necessary.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    res = model.load_state_dict(new_state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        logger.warning("State dict mismatch — missing: %s | unexpected: %s",
                       res.missing_keys, res.unexpected_keys)

    # Set device and move model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set model to evaluation mode.
    logger.info("Loaded model %s on device %s", model.__class__.__name__, device)
        
    model_name = config.get("model_name", "defaultModel")
    config_name = os.path.basename(config_path).split('.')[0]

    # Prepare testing data.
    # train.py runs argparse at import time — shield our argv from it.
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from train import prepare_testing_data
    finally:
        sys.argv = _argv
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
        config=config,  # needed for mean/std in LIME's perturbation normalization
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
