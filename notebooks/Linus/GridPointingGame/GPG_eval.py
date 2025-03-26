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
import yaml
import pickle
import random
from PIL import Image
from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  
# from GRADCAM_eval import GradCamEvaluator  # Uncomment if implemented.
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

# set model path and additional arguments
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/resnet34_bcos_v2_minimal.yaml")
ADDITIONAL_ARGS = {
    "test_batchSize": 12
}


#setpup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model(config):
    """Load and return the model using the DETECTOR registry."""
    logger.info("Registered models: %s", list(DETECTOR.data.keys()))
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    return model

def load_config(path, additional_args={}):
    """Load and merge configuration files with any additional overrides."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    try:
        with open('./training/config/test_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/test_config.yaml'), 'r') as f:
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

def get_images_from_dataloader(data_loader):
    """Extract and return all images from a DataLoader."""
    all_images = []
    for batch in data_loader:
        images = batch['image']  # shape: [batch_size, C, H, W]
        for img in images:
            all_images.append(img)
    return all_images

def preprocess_image(img):
    # img is expected to be a tensor of shape [B, C, H, W] in a batch.
    if img.shape[1] == 3:
        img = torch.cat([img, 1.0 - img], dim=1)
    return img

class Analyser:

    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        overall, raw = self.analysis()
        self.save_results(raw, overall)

    def save_results(self, raw, overall):
        """Save raw and overall results to disk."""
        save_folder = os.path.join("results", str(self.grid_split), f"{self.model_name}_{self.weights_name}")
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, "results.pkl"), "wb") as f:
            pickle.dump(raw, f)
        with open(os.path.join(save_folder, "overall.pkl"), "wb") as f:
            pickle.dump(overall, f)
        logger.info("Results saved to folder: %s", save_folder)
        
    def load_results(self, load_overall=True):
        """Load and print a summary of previously saved results."""
        load_folder = os.path.join("results", str(self.grid_split), f"{self.model_name}_{self.weights_name}")
        file_path = os.path.join(load_folder, "overall.pkl" if load_overall else "results.pkl")
        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
        logger.info("Results loaded from %s", file_path)
        if load_overall:
            localisation_metric = loaded.get("localisation_metric", None)
            percentiles = loaded.get("percentiles", None)
            if localisation_metric is not None and percentiles is not None:
                logger.info("Overall results: %d evaluations, percentiles: %s",
                            len(localisation_metric), percentiles)
            else:
                logger.warning("Overall results missing expected keys.")
        else:
            sorted_results = sorted(loaded, key=lambda res: res.get("accuracy", 0), reverse=True)
            logger.info("Top raw results:")
            for idx, res in enumerate(sorted_results[:10]): #10?
                logger.info("[%d] %s - Accuracy: %s", idx + 1, res.get("path", "N/A"), res.get("accuracy", "N/A"))
        return loaded

class RankedGPGCreator(Analyser):
    def __init__(self, base_output_dir, grid_size=(3, 3),xai_method=None,  max_grids=3, plotting_only=False,
                 model=None, model_name="default", weights_name="default",
                 loader=None, dataset=None, device=None, config=None, grid_split=3):
        """
        Initializes the grid creator.
        
        Args:
            base_output_dir (str): Base directory to save grids.
            grid_size (tuple): Grid dimensions.
            xai_method (str): Evaluation method to use.
            max_grids (int): Maximum number of grids to create.
            plotting_only (bool): If True, load existing results without grid creation.
            model, loader, dataset, device: Required objects for grid creation and evaluation.
            grid_split (int): Grid split parameter.
        """
        self.grid_size = grid_size
        self.xai_method = xai_method
        self.max_grids = max_grids
        self.model = model
        self.loader = loader
        self.dataset = dataset
        self.model_name = model_name
        self.weights_name = weights_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{weights_name}")
        self.device = device
        self.grid_split = grid_split


        if plotting_only:
            self.load_results()
            return
        
        self.output_dir = os.path.join(self.output_folder, str(self.grid_split))
        os.makedirs(self.output_dir, exist_ok=True)

        # Process an load images ranking.
        self.ranking_file = os.path.join(self.output_folder, "sorted_confs.pkl")
        if os.path.exists(self.ranking_file):
            self.sorted_confs = self.load_ranking(self.ranking_file)
            logger.info("Loaded existing sorted confidences from %s", self.ranking_file)
        else:
            self.sorted_confs = self.compute_sorted_confs()
            self.save_ranking(self.sorted_confs, self.ranking_file)
            logger.info("Saved sorted confidences to %s", self.ranking_file)
            
    def compute_sorted_confs(self):
        """Compute and return image ranking based on model confidence."""
        ranking = {0: [], 1: []}
        img_idx = 0
        with torch.no_grad():
            for batch in self.loader:
                images = batch['image'].to(self.device)
                if self.xai_method == "bcos":
                    images = preprocess_image(images)
                labels = batch['label'].to(self.device)
                output = self.model({'image': images, 'label': labels})
                logits = output['cls']
                for i in range(len(images)):
                    true_label = int(labels[i].item())
                    predicted_label = logits[i].argmax().item()
                    if true_label == 0 and predicted_label == true_label:
                        confidence = logits[i, predicted_label].item()
                        ranking[0].append((img_idx, confidence))
                    elif true_label == 1 and predicted_label == true_label:
                        confidence = logits[i, predicted_label].item()
                        ranking[1].append((img_idx, confidence))
                    img_idx += 1
        
        for cls in ranking:
            ranking[cls] = sorted(ranking[cls], key=lambda x: x[1], reverse=True)
            logger.debug("Class %d: %d images after sorting.", cls, len(ranking[cls]))
        return ranking
        
    '''
    def get_sorted_indices(self):
        """Select top image indices based on rankings for grid creation."""
        k = self.max_grids
        indices = {}
        for cls in [0, 1]:
            cls_list = self.sorted_confs.get(cls, [])
            required = k * (self.grid_size[0] * self.grid_size[1] - 1) if cls == 0 else k
            indices[cls] = [idx for idx, _ in cls_list[:required]]
        return indices
    '''

    def get_image_by_index(self, idx):
        """Return an image tensor from the dataset based on its index."""
        if self.dataset is None:
            raise ValueError("Dataset not provided for image retrieval.")
        image, _, _, _ = self.dataset[idx]
        return image
    
    def save_ranking(self, ranking, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(ranking, f)

    def load_ranking(self, file_path):
        with open(file_path, "rb") as f:
            ranking = pickle.load(f)
        return ranking

    def analysis(self):
        """Evaluate pre-created grids and compute overall metrics."""
        results_folder = os.path.join("results", str(self.grid_split), f"{self.model_name}_{self.weights_name}")
        raw_results_file = os.path.join(results_folder, "results.pkl")
        if os.path.exists(raw_results_file):
            raise RuntimeError(f"Results already exist at {raw_results_file}. Use load_results() instead.")

        grid_dir = os.path.join(self.output_folder, str(self.grid_split))
        grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]
        logger.info("Found %d grid tensors in %s.", len(grid_paths), grid_dir)

        preprocessed_tensors = [torch.load(path, map_location=self.device) for path in grid_paths]
        logger.info("Loaded all grid tensors.")

        # Instantiate the evaluator based on the selected XAI method.
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
        
        # Check for existing files
        existing_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pt')]
        logger.debug("Found %d existing .pt files in %s.", len(existing_files), self.output_dir)
        
        if len(existing_files) >= self.max_grids:
            logger.info("Existing grid files found. Skipping grid creation.")
            return
        
        # Copy the ranked real/fake lists
        ranked_real = self.sorted_confs.get(0, []).copy()
        ranked_fake = self.sorted_confs.get(1, []).copy()
        logger.debug("Ranked real size: %d, Ranked fake size: %d", len(ranked_real), len(ranked_fake))
        
        n_imgs = self.grid_size[0] * self.grid_size[1]
        logger.debug("Grid size: %s, total images per grid: %d", self.grid_size, n_imgs)
        side = int(np.sqrt(n_imgs))
        logger.debug("Calculated side length: %d", side)
        
        grid_count = 0
        
        while grid_count < self.max_grids:
            logger.info("--- Creating grid %d of %d ---", grid_count + 1, self.max_grids)
            required_real = n_imgs - 1
            fake_count = len(ranked_fake)
            real_count = len(ranked_real)
            logger.debug("Need %d real images, have %d. Need 1 fake image, have %d.",
                         required_real, real_count, fake_count)
            
            if fake_count < 1 or real_count < required_real:
                logger.warning(
                    "Not enough images to create more grids. Fake count: %d, Real count: %d (required %d real images)",
                    fake_count, real_count, required_real
                )
                break
            
            # Pop the first fake image tuple
            fake_tuple = ranked_fake.pop(0)  # tuple: (img_idx, confidence)
            logger.info("Selected fake image: idx %d with confidence %.4f", fake_tuple[0], fake_tuple[1])
            
            # Retrieve the fake image tensor
            fake_img = self.get_image_by_index(fake_tuple[0])
            logger.debug("Fake image retrieved, shape: %s", fake_img.shape if hasattr(fake_img, 'shape') else "N/A")
            
            # For real images, take the first (n_imgs - 1) tuples
            selected_real_tuples = ranked_real[:required_real]
            logger.info("Selected real images: %s", selected_real_tuples)
            
            # Remove them from the list
            ranked_real = ranked_real[required_real:]
            
            # Retrieve the real images
            selected_real = [self.get_image_by_index(idx) for idx, conf in selected_real_tuples]
            logger.debug("Retrieved %d real images.", len(selected_real))
            
            # Combine the selected real images and the fake image
            images = selected_real + [fake_img]
            logger.debug("Combined real + fake images. Total: %d", len(images))
            
            random.shuffle(images)
            logger.debug("Shuffled images.")
            
            # Find the fake image's index in the shuffled list
            fake_index = next(i for i, img in enumerate(images) if torch.equal(img, fake_img))
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            logger.debug("Fake image is at shuffled index %d, final index %d in the grid.", fake_index, final_fake_index)
            
            # Stack images and reshape into a grid tensor
            stacked = torch.stack(images, dim=0)
            logger.debug("Stacked images, shape: %s", stacked.shape)
            
            grid_tensor = (
                stacked.view(-1, side, side, *stacked.shape[-3:])
                      .permute(0, 3, 2, 4, 1, 5)
                      .reshape(-1, stacked.shape[1], stacked.shape[2] * side, stacked.shape[3] * side)
            )
            logger.debug("Reshaped grid tensor, final shape: %s", grid_tensor.shape)
            
            # Save the grid
            base_name = f"grid_{grid_count}_fake_{final_fake_index}.pt"
            path_to_grid = os.path.join(self.output_dir, base_name)
            torch.save(grid_tensor, path_to_grid)
            logger.info("Saved grid tensor: %s", path_to_grid)
            
            grid_count += 1
        
        logger.info("=== Finished GPG grid creation. Created %d grids. ===", grid_count)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    parser.add_argument("--base_output_dir", type=str, default="datasets/GPG_grids",
                        help="Base output directory for grids.")
    parser.add_argument("--max_grids", type=int, default=2, help="Maximum number of grids to create.")
    parser.add_argument("--xai_method", type=str, default="bcos",
                        choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to model configuration file.")
    parser.add_argument("--grid_split", type=int, default=3,
                        help="Grid size for evaluation (e.g. 3 for a 3x3 grid).")
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
    # Remove "module." prefix if necessary.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(next(model.parameters()).device)

    model.eval()

    logger.info("Loaded model %s on device %s", model.__class__.__name__, device)
        
    model_name = config.get("model_name", "defaultModel")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Weight file not found: {pretrained_path}")
    weights_name = os.path.basename(pretrained_path).split('.')[0]

    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = list(test_data_loaders.values())[0]

    grid_creator = RankedGPGCreator(
        base_output_dir=args.base_output_dir,
        grid_size=grid_size,
        xai_method=args.xai_method,
        max_grids=args.max_grids,
        model=model,
        model_name=model_name,
        weights_name=weights_name,
        loader=test_loader,
        dataset=test_loader.dataset,
        device=device,
        grid_split=args.grid_split
    )

    grid_creator.create_GPG_grids()
    grid_creator.run()
    grid_creator.load_results(load_overall=False)

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py