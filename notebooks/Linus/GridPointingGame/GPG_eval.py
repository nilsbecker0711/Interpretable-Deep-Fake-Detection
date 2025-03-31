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
# from GRADCAM_eval import GradCamEvaluator  # Uncomment if implemented.

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

# set model path and additional arguments
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_minimal.yaml")
ADDITIONAL_ARGS = {
    "test_batchSize": 12
}

#setpup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GridPointingGameCreator(Analyser):
    def __init__(self, base_output_dir, grid_size=(3, 3), xai_method=None, max_grids=3, plotting_only=False,
                 model=None, model_name="default", weights_name="default",
                 test_data_loaders=None, dataset=None, device=None, config=None, grid_split=3):
        """
        Initialize grid creator with specified parameters.
        base_output_dir: Base directory for grids.
        grid_size: Dimensions of the grid (e.g., (3,3)).
        xai_method: "bcos", "lime", or "gradcam".
        max_grids: Maximum number of grids to create.
        plotting_only: If True, load existing results.
        """
        self.grid_size = grid_size
        self.xai_method = xai_method
        self.max_grids = max_grids
        self.model = model
        self.test_data_loaders = test_data_loaders
        self.dataset = dataset
        self.model_name = model_name
        self.weights_name = weights_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{weights_name}")
        self.device = device
        self.grid_split = grid_split

        if plotting_only:
            self.load_results()
            return
        
        # Create output directory for grids.
        self.output_dir = os.path.join(self.output_folder, str(self.grid_split))
        os.makedirs(self.output_dir, exist_ok=True)

        # Load or compute sorted image rankings.
        self.ranking_file = os.path.join(self.output_folder, "sorted_confs.pkl")
        if os.path.exists(self.ranking_file):
            self.sorted_confs = self.load_ranking(self.ranking_file)
            logger.info("Loaded sorted confidences from %s", self.ranking_file)
        else:
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
            # Process each image in the batch.
            for j in range(num_samples):
                image = img_batch[j].unsqueeze(0)  # shape: [1, C, H, W]
                label = label_batch[j]
                true_label = int(label.item())  # Convert label tensor to int.
                image_path = path_of_image[j]
                
                if self.xai_method == "bcos":
                    image = preprocess_image(image)
                if self.xai_method == "lime":
                    image = image[:,:3]
                output = self.model({'image': image, 'label': label})
                logit = output['cls']  # Expected shape: [1, num_classes]
                # Get predicted label from the first (and only) sample.
                predicted_label = logit[0].argmax().item()
                # Compute confidence from the corresponding logit.
                confidence = logit[0, predicted_label].item()
                
                # Check for class 1.
                if true_label == 1:
                    if predicted_label == 1:
                        ranking[1].append((image_path, confidence, true_label))
                # Check for class 0.
                if true_label == 0:
                    if predicted_label == 0:
                        ranking[0].append((image_path, confidence, true_label))
        
        # Sort each class's ranking by descending confidence.
        for cls in ranking:
            ranking[cls] = sorted(ranking[cls], key=lambda x: x[1], reverse=True)
            logger.debug("Class %d: %d images after sorting.", cls, len(ranking[cls]))
        return ranking
    
    def get_sorted_image_paths(self):
        """Select top image indices based on rankings for grid creation,
        filtering each tuple by a confidence threshold (sigmoid(confidence) > 0.5).
        For class 0 (real), selects k * (grid_size[0] * grid_size[1] - 1) images,
        and for class 1 (fake), selects k images.
        """
        # Helper function: returns True if sigmoid(confidence) > 0.5.
        def get_conf_mask_v(tup):
            # tup is (img_idx, confidence)
            return torch.tensor(tup[1]).sigmoid().item() > 0.5
    
        k = self.max_grids
        sorted_image_paths = {}
        for cls in [0, 1]:
            # Get the sorted list for this class (list of tuples: (img_idx, confidence))
            cls_list = self.sorted_confs.get(cls, [])
            # Filter the list by confidence threshold.
            filtered = [tup for tup in cls_list if get_conf_mask_v(tup)]
            # Determine the number of required images:
            required = k * (self.grid_size[0] * self.grid_size[1] - 1) if cls == 0 else k
            # Select only the image indices from the filtered list (up to the required number).
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
        results_folder = os.path.join("results", str(self.grid_split), f"{self.model_name}_{self.weights_name}")
        raw_results_file = os.path.join(results_folder, "results.pkl")
        if os.path.exists(raw_results_file):
            raise RuntimeError(f"Results already exist at {raw_results_file}. Use load_results() instead.")

        # List grid tensor files.
        grid_dir = os.path.join(self.output_folder, str(self.grid_split))
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]
        logger.info("Found %d grid tensors in %s.", len(grid_paths), grid_dir)

        # Load each grid tensor.
        preprocessed_tensors = [torch.load(path, map_location=self.device) for path in grid_paths]
        logger.info("Loaded all grid tensors.")

        # Choose evaluator based on xai_method.
        if self.xai_method == "bcos":
            evaluator = BCOSEvaluator(self.model, self.device)
        elif self.xai_method == "lime":
            evaluator = LIMEEvaluator(self.model, self.device)
        elif self.xai_method == "gradcam":
            raise NotImplementedError("GradCAM evaluator not implemented.")
        else:
            raise ValueError(f"Unknown xai_method: {self.xai_method}")

        raw_results = evaluator.evaluate(preprocessed_tensors, grid_paths, self.grid_split)
        grid_accuracies = [res["accuracy"] for res in raw_results]
        percentiles = np.percentile(np.array(grid_accuracies), [25, 50, 75, 100])
        logger.info("Localisation accuracy percentiles: %s", percentiles)
        overall = {"localisation_metric": grid_accuracies, "percentiles": percentiles}
        return overall, raw_results

    def create_GPG_grids(self):
        """Create grids by combining ranked real and fake images."""
        logger.info("=== Starting GPG grid creation in %s ===", self.output_folder)
        
        # Check if grids already exist.
        existing_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pt')]
        logger.debug("Found %d existing .pt files in %s.", len(existing_files), self.output_dir)
        if len(existing_files) >= self.max_grids:
            logger.info("Existing grid files found. Skipping grid creation.")
            return
        
        # Copy the ranked real/fake lists
        #ranked_real = self.sorted_confs.get(0, []).copy()
        #ranked_fake = self.sorted_confs.get(1, []).copy()

        # Get sorted image paths (tuples of (image_path, confidence, label))
        sorted_image_paths = self.get_sorted_image_paths()
        # Expect one fake (class 1) and remaining real (class 0) images.
        ranked_real = sorted_image_paths.get(0, []).copy()
        ranked_fake = sorted_image_paths.get(1, []).copy()
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
            logger.info("Selected fake image: %s with confidence %.4f", fake_tuple[0], fake_tuple[1])
            expected_label = 1
            fake_img = self.load_sample_by_path(fake_tuple[0], expected_label)
            if self.xai_method == "lime":
                fake_img = fake_img[:3]
            logger.debug("Fake image shape: %s", fake_img.shape if hasattr(fake_img, 'shape') else "N/A")
            
            # Select first required_real real image tuples.
            selected_real_tuples = ranked_real[:required_real]
            logger.info("Selected real image paths: %s", selected_real_tuples)
            ranked_real = ranked_real[required_real:]  # Remove used entries.
            
            # Retrieve real images using load_sample_by_path for consistency.
            expected_label = 0
            selected_real = [self.load_sample_by_path(img_path, expected_label) for img_path, _, _ in selected_real_tuples]
            if self.xai_method == "lime":
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
            base_name = f"grid_{grid_count}_fake_{final_fake_index}.pt"
            path_to_grid = os.path.join(self.output_dir, base_name)
            torch.save(grid_tensor, path_to_grid)
            logger.info("Saved grid tensor: %s", path_to_grid)
            
            grid_count += 1
        
        logger.info("=== Finished grid creation. Created %d grids. ===", grid_count)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate grids using a model and XAI method.")
    parser.add_argument("--base_output_dir", type=str, default="datasets/GPG_grids",
                        help="Base output directory for grids.")
    parser.add_argument("--max_grids", type=int, default=2, help="Max number of grids to create.")
    parser.add_argument("--xai_method", type=str, default="bcos",
                        choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to model configuration file.")
    parser.add_argument("--grid_split", type=int, default=3,
                        help="Grid split (e.g., 3 for a 3x3 grid).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    grid_size = (args.grid_split, args.grid_split)
    logger.info("Parameters: XAI=%s, Base=%s, Model=%s, Grid=%dx%d",
                args.xai_method, args.base_output_dir, args.model_path, args.grid_split, args.grid_split)

    config = load_config(args.model_path, additional_args=ADDITIONAL_ARGS)
    model = load_model(config)
    
    pretrained_path = config['pretrained']
    state_dict = torch.load(pretrained_path)
    # Remove "module." prefix from state_dict keys if necessary.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    
    # Set device and move model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set model to evaluation mode.
    logger.info("Loaded model %s on device %s", model.__class__.__name__, device)
        
    model_name = config.get("model_name", "defaultModel")
    weights_name = os.path.basename(pretrained_path).split('.')[0]

    # Prepare testing data.
    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = list(test_data_loaders.values())[0]
    dataset = test_loader.dataset

    # Initialize grid creator with all required objects.
    grid_creator = GridPointingGameCreator(
        base_output_dir=args.base_output_dir,
        grid_size=grid_size,
        xai_method=args.xai_method,
        max_grids=args.max_grids,
        model=model,
        model_name=model_name,
        weights_name=weights_name,
        test_data_loaders= test_data_loaders,
        dataset=dataset,
        device=device,
        grid_split=args.grid_split
    )

    grid_creator.create_GPG_grids()  # Create new grids.
    grid_creator.run()               # Run analysis.
    grid_creator.load_results(load_overall=False)  # Load and display results.

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py