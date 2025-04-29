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
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset



#######################
# set model path, config path and additional arguments

#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_bcos_res_2_5_config.yaml")
#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_bcos_res_1_25_config.yaml")
#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_res_lime_config.yaml")
CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_res_gradcam_config.yaml")
#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_res_xgrad_config.yaml")
#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_res_layergrad_config.yaml")
#CONFIG_PATH = os.path.join(PROJECT_PATH, "results/test_res_grad++_config.yaml")

#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_2_5_best_hpo.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_1_25_best_hpo.yaml")
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34.yaml")
#MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet_bcos_minimal.yaml")

ADDITIONAL_ARGS = {
    "test_batchSize": 20
}
#######################



#setpup logginglogging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GridPointingGameCreator(Analyser):
    def __init__(self, base_output_dir, grid_size=(3, 3), xai_method=None, max_grids=3,
                 model=None, model_name="default", config_name="default",
                 test_data_loaders=None, dataset=None, device=None, config=None, grid_split=3, overwrite=False, quantitativ=False, threshold_steps=0, b_value_name=0):
        """

        """
        self.grid_size = grid_size
        self.xai_method = xai_method
        self.max_grids = max_grids
        self.model = model
        self.test_data_loaders = test_data_loaders
        self.dataset = dataset
        self.model_name = model_name
        self.config_name = config_name
        self.device = device
        self.grid_split = grid_split
        self.overwrite = overwrite
        self.quantitativ = quantitativ
        self.threshold_steps = threshold_steps
        self.b_value_name = b_value_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{config_name}")
        self.confidence_dir= os.path.join(base_output_dir, f"{model_name}_{b_value_name}")
        self.grid_dir = os.path.join(base_output_dir, f"{model_name}_{b_value_name}", f"{grid_size[0]}x{grid_size[1]}")
        self.results_dir = os.path.join(self.output_folder, f"{grid_size[0]}x{grid_size[1]}")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.confidence_dir, exist_ok=True)
        os.makedirs(self.grid_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        

        # Load or compute sorted image rankings.
        self.ranking_file = os.path.join(self.confidence_dir, "sorted_confs.pkl")

        if os.path.exists(self.ranking_file) and not self.overwrite:
            self.sorted_confs = self.load_ranking(self.ranking_file)
            logger.info("Loaded sorted confidences from %s", self.ranking_file)
        else:
            if self.overwrite and os.path.exists(self.ranking_file):
                logger.info("Overwrite is enabled. Recomputing and replacing %s", self.ranking_file)
            self.sorted_confs = self.compute_sorted_confs()
            self.save_ranking(self.sorted_confs, self.ranking_file)
            logger.info("Saved sorted confidences to %s", self.ranking_file)

    def compute_sorted_confs(self):
        """Compute ranking by storing (image_path, confidence, label) for each correctly classified image."""
        ranking = {0: [], 1: []}
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
    
            # Remove extra key if present.
            data_dict.pop('label_spe', None)
            # Convert labels to binary.
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
    
            num_samples = img_batch.shape[0]
            for j in range(num_samples):
                image = img_batch[j].unsqueeze(0)  # shape: [1, C, H, W]
                label = label_batch[j]
                true_label = int(label.item())
                image_path = path_of_image[j]
    
                if self.xai_method == "bcos":
                    image = preprocess_image(image)
                if self.xai_method in ["lime", "gradcam", "xgrad", "grad++", "layergrad"]:
                    image = image[:, :3]
    
                output = self.model({'image': image, 'label': label})
                logit = output['cls']  # Expected shape: [1, num_classes]
                predicted_label = logit[0].argmax().item()
    
                # Compute confidence using softmax
                probabilities = torch.nn.functional.softmax(logit, dim=1)
                confidence = probabilities[0, predicted_label].item()
    
                # Only store if prediction is correct
                if true_label == predicted_label:
                    ranking[true_label].append((image_path, confidence, true_label))
    
        # Sort each class's ranking by descending confidence.
        for cls in ranking:
            ranking[cls] = sorted(ranking[cls], key=lambda x: x[1], reverse=True)
            logger.debug("Class %d: %d images after sorting.", cls, len(ranking[cls]))
        return ranking
    
    def get_sorted_image_paths(self):
        """Select top image indices based on rankings for grid creation,
        filtering each tuple by a confidence threshold (confidence > 0.5).
        For class 0 (real), selects k * (grid_size[0] * grid_size[1] - 1) images,
        and for class 1 (fake), selects k images.
        """
        # Helper function: returns True if confidence > 0.5 (confidence is already from softmax)
        def get_conf_mask_v(tup):
            return tup[1] > 0.5
    
        k = self.max_grids
        sorted_image_paths = {}
        for cls in [0, 1]:
            cls_list = self.sorted_confs.get(cls, [])
            filtered = [tup for tup in cls_list if get_conf_mask_v(tup)]
            print(len(filtered))
            required = k * (self.grid_size[0] * self.grid_size[1] - 1) if cls == 0 else k
            sorted_image_paths[cls] = filtered[:required]
        return sorted_image_paths

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
        return sample[0]
    
    def save_ranking(self, ranking, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(ranking, f)

    def load_ranking(self, file_path):
        with open(file_path, "rb") as f:
            ranking = pickle.load(f)
        return ranking
    
    def analysis(self):
        """Evaluate grid tensors and compute overall metrics."""
        raw_results_file = os.path.join(self.results_dir, "results.pkl")

        if os.path.exists(raw_results_file) and not self.overwrite:
            raise RuntimeError(f"Results already exist at {raw_results_file}. Use those results or set overwrite=True.")

        if self.overwrite and os.path.exists(raw_results_file):
            logger.info("Overwrite is enabled. Existing results at %s will be overwritten.", raw_results_file)

        # List grid tensor files.
        grid_paths = [os.path.join(self.grid_dir, f) for f in os.listdir(self.grid_dir) if f.endswith('.pt')]
        logger.info("Found %d grid tensors in %s.", len(grid_paths), self.grid_dir)

        # Load each grid tensor.
        preprocessed_tensors = [torch.load(path, map_location=self.device) for path in grid_paths]
        logger.info("Loaded all grid tensors.")

        # Choose evaluator based on xai_method.
        if self.xai_method == "bcos":
            evaluator = BCOSEvaluator(self.model, self.device)
        elif self.xai_method == "lime":
            evaluator = LIMEEvaluator(self.model, self.device)
        elif self.xai_method in ["gradcam", "xgrad", "grad++", "layergrad"]:
            evaluator = GradCamEvaluator(self.model, self.device, method=self.xai_method)
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")

        # Run evaluation with thresholding
        raw_results = evaluator.evaluate(preprocessed_tensors, grid_paths, self.grid_split, threshold_steps=self.threshold_steps)

        return raw_results

    def create_GPG_grids(self):
        """Create grids by combining ranked real and fake images."""
        logger.info("=== Starting GPG grid creation in %s ===", self.output_folder)
        random.seed(32) 
        
        # Check if grids already exist.
        existing_files = [f for f in os.listdir(self.grid_dir) if f.endswith('.pt')]
        logger.debug("Found %d existing .pt files in %s.", len(existing_files), self.grid_dir)

        if existing_files and self.overwrite:
            logger.info("Overwrite is enabled. Deleting existing grid files and continue with creatring new grids.")
            for f in existing_files:
                os.remove(os.path.join(self.grid_dir, f))
            existing_files = []  # Reset list after deletion

        if len(existing_files) >= self.max_grids:
            logger.info("Enough grid files in folder. Skipping grid creation.")
            return
        else:
            for f in existing_files:
                os.remove(os.path.join(self.grid_dir, f))
            existing_files = []  # Reset list after deletion
            

        # Get sorted image paths (tuples of (image_path, confidence, label))
        sorted_image_paths = self.get_sorted_image_paths()
        # Expect one fake (class 1) and remaining real (class 0) images.
        ranked_real = sorted_image_paths.get(0, []).copy()
        ranked_fake = sorted_image_paths.get(1, []).copy()

        # wenn doch nicht das in if rein setzen
        random.shuffle(ranked_real)

        if self.quantitativ:
            random.shuffle(ranked_fake)

        logger.debug("Ranked real: %d, Ranked fake: %d", len(ranked_real), len(ranked_fake))
        
        n_imgs = self.grid_size[0] * self.grid_size[1]
        logger.debug("Total images per grid: %d", n_imgs)
        side = int(np.sqrt(n_imgs))
        logger.debug("Calculated grid side length: %d", side)
        
        grid_count = 0
        while grid_count < self.max_grids:
            logger.info("--- Creating grid %d of %d ---", grid_count + 1, self.max_grids)
            required_real = n_imgs - 1  # Reserve 1 slot for fake image.
            fake_count = len(ranked_fake)
            real_count = len(ranked_real)
            logger.debug("Need %d real, have %d; need 1 fake, have %d.", required_real, real_count, fake_count)
            
            if fake_count < 1 or real_count < required_real:
                logger.warning("Not enough images: fake %d, real %d (required %d)", fake_count, real_count, required_real)
                break

            # Get first fake tuple and remove it.
            fake_tuple = ranked_fake.pop(0)
            logger.info("Selected fake image: %s with confidence %.4f", fake_tuple[0], torch.tensor(fake_tuple[1]).sigmoid().item())
            expected_label = 1
            fake_img = self.load_sample_by_path(fake_tuple[0], expected_label)
            if self.xai_method in ["lime", "gradcam", "xgrad", "grad++", "layergrad"]:
                fake_img = fake_img[:3]
            logger.debug("Fake image shape: %s", fake_img.shape if hasattr(fake_img, 'shape') else "N/A")
            
            # Select first required_real real image tuples.
            selected_real_tuples = ranked_real[:required_real]
            logger.info("Selected real image paths: %s", selected_real_tuples)
            ranked_real = ranked_real[required_real:]  # Remove used entries.
            
            # Retrieve real images using load_sample_by_path for consistency.
            expected_label = 0
            selected_real = [self.load_sample_by_path(img_path, expected_label) for img_path, _, _ in selected_real_tuples]
            if self.xai_method in ["lime", "gradcam", "xgrad", "grad++", "layergrad"]:
                selected_real = [img[:3] for img in selected_real]
                
            logger.debug("Retrieved %d real images.", len(selected_real))
            
            # Combine real and fake images.
            images = selected_real + [fake_img]
            logger.debug("Combined image count: %d", len(images))
            random.shuffle(images)  # Shuffle grid placement.
            logger.debug("Images shuffled.")
            
            # Find fake image index in shuffled list.
            fake_index = next(i for i, img in enumerate(images) if torch.equal(img, fake_img))
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            logger.debug("Fake image: shuffled index %d, final index %d", fake_index, final_fake_index)
            
            # Stack images and reshape into grid tensor.
            stacked = torch.stack(images, dim=0)
            logger.debug("Stacked images shape: %s", stacked.shape)
            grid_tensor = (
                stacked.view(-1, side, side, *stacked.shape[-3:])
                       .permute(0, 3, 2, 4, 1, 5)
                       .reshape(-1, stacked.shape[1], stacked.shape[2] * side, stacked.shape[3] * side)
            )
            logger.debug("Grid tensor shape: %s", grid_tensor.shape)
            
            # Save grid tensor with fake position encoded in filename.
base_name = f"{self.model_name}_{self.b_value_name}_grid_{grid_count}_fake_{final_fake_index}_conf_fake{fake_tuple[1]:.4f}.pt"
            path_to_grid = os.path.join(self.grid_dir, base_name)
            torch.save(grid_tensor, path_to_grid)
            logger.info("Saved grid tensor: %s", path_to_grid)
            
            grid_count += 1
        
        logger.info("=== Finished grid creation. Created %d grids. ===", grid_count)


def main():
    config = load_config(MODEL_PATH, CONFIG_PATH, additional_args=ADDITIONAL_ARGS)

    required_keys = ["grid_split", "overwrite", "quantitativ", "xai_method", "max_grids"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    logger.info("Parameters: XAI=%s, Base=%s, Model=%s, Grid=%dx%d", config['xai_method'], config['base_output_dir'], MODEL_PATH, config['grid_split'], config['grid_split'])

    model = load_model(config)
    
    grid_size = (config['grid_split'], config['grid_split'])

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
        
    model_name = config.get("model_name", "defaultModel")
    config_name = os.path.basename(CONFIG_PATH).split('.')[0]
    b_value_name = str(config.get("backbone_config", {}).get("b", "default")).replace(".", "_")
    
    # Prepare testing data.
    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = list(test_data_loaders.values())[0]
    dataset = test_loader.dataset

    # Initialize grid creator with all required objects.
    grid_creator = GridPointingGameCreator(
        base_output_dir=config.get("base_output_dir", "results"),
        grid_size=grid_size,
        xai_method=config["xai_method"],
        max_grids=config["max_grids"],
        model=model,
        model_name=model_name,
        config_name=config_name,
        test_data_loaders=test_data_loaders,
        dataset=dataset,
        device=device,
        grid_split=config["grid_split"],
        overwrite=config["overwrite"],
        quantitativ=config["quantitativ"],
        threshold_steps=config["threshold_steps"],
        b_value_name=b_value_name
    )

    grid_creator.create_GPG_grids()  # Create new grids.
    grid_creator.run()               # Run analysis.

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py

