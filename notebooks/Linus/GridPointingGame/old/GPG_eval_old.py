import os
import sys

# Absoluter Pfad zum Projektstamm (hier zwei Ebenen höher, da sich GPG_eval.py in notebooks/Linus/GridPointingGame befindet)
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

import torch
import argparse
import numpy as np
import yaml
import pickle
import random
from PIL import Image

# Evaluators.
#from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  # Adjust the import path if needed.
# from GRADCAM_eval import GradCamEvaluator  # Uncomment if available.

from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

XAI_METHOD = "lime"
BASE_OUTPUT_DIR = "datasets/GPG_grids"
GRID_SPLIT = 3
MAX_GRIDS = 2
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception.yaml")

ADDITIONAL_ARGS = {
    "model_name": "xception",
    "test_batchSize": 12,
}

def load_model(config):
    """
    Load and return the model using the DETECTOR registry.
    """
    print("[DEBUG] Registered models:", DETECTOR.data.keys())
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    return model

def load_config(path, additional_args={}):
    # Parse primary config.
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Attempt to load training config.
    try:
        with open('./training/config/test_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/test_config.yaml'), 'r') as f:
            config2 = yaml.safe_load(f)
    
    # If label_dict is present in the first config, update the training config.
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    
    config.update(config2)
    
    if config.get('dry_run', False):
        config['nEpochs'] = 0
        config['save_feat'] = False
        
    # Override with any additional arguments.
    for key, value in additional_args.items():
        config[key] = value
    return config

def get_images_from_dataloader(data_loader):
    """
    Lädt alle Bilder aus dem DataLoader und gibt eine Liste von Bild-Tensoren zurück.
    Alle Tensoren bleiben auf der CPU.
    """
    all_images = []
    for batch in data_loader:
        images = batch['image']  # shape: [batch_size, C, H, W]
        for img in images:
            all_images.append(img)
    return all_images

class RankedGPGCreator:
    def __init__(self, base_output_dir, grid_size=(3, 3), max_grids=2, plotting_only=False,
                 model=None, model_name="default", weights_name="default",
                 loader=None, dataset=None):
        """
        Args:
            base_output_dir (str): Base directory in which to save grids.
            grid_size (tuple): Dimensions of the grid (rows, columns).
            max_grids (int): Maximum number of grids to create.
            model (callable): A model that takes an image tensor and returns a confidence score.
            model_name (str): Name of the model (used for naming the folder).
            weights_name (str): Name of the weights (used for naming the folder).
            loader (DataLoader, optional): Falls übergeben, werden reale Bilder aus dem DataLoader geladen.
        """
        self.grid_size = grid_size
        self.max_grids = max_grids
        self.model = model
        self.loader = loader
        self.dataset = dataset  # Store the dataset instance here.
        self.model_name = model_name
        self.weights_name = weights_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{weights_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create subdirectories for 3-channel.
        self.output_dir_3ch = os.path.join(self.output_folder, "3ch")
        os.makedirs(self.output_dir_3ch, exist_ok=True)

        # Process images ranking.
        self.ranking_file = os.path.join(self.output_folder, "sorted_confs.pkl")
        if os.path.exists(self.ranking_file):
            self.sorted_confs = self.load_ranking(self.ranking_file)
            print(f"[DEBUG] Loaded existing sorted confidences from {self.ranking_file}")
        else:
            self.sorted_confs = self.compute_sorted_confs()
            self.save_ranking(self.sorted_confs, self.ranking_file)
            print(f"[DEBUG] Saved sorted confidences to {self.ranking_file}")

    def compute_sorted_confs(self):
        """
        Berechnet und speichert Bildindizes sortiert nach der Klassifikationskonfidenz
        für beide Klassen (0: real, 1: fake). Es werden nur (Bildindex, confidence)-Tupel gespeichert.
        """
        ranking = {0: [], 1: []}
        # Wir gehen davon aus, dass der loader beide Klassen enthält.
        loader = self.loader  
        img_idx = 0
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)  # [batch_size, C, H, W]
                labels = batch['label'].to(self.device)    # Erwartet als Integer-Tensor
                output = self.model({'image': images, 'label': labels})
                logits = output['cls']
                # Für jedes Bild in der Batch:
                for i in range(len(images)):
                    true_label = int(labels[i].item())
                    predicted_label = logits[i].argmax().item()

                    """
                    # Nur korrekt klassifizierte Bilder berücksichtigen:
                    if predicted_label == true_label:
                        confidence = logits[i, predicted_label].item()
                        ranking[true_label].append((img_idx, confidence))
                    img_idx += 1
                    """
                    
                    # For real images, include only correctly classified ones.
                    if true_label == 0:
                        if predicted_label == true_label:
                            confidence = logits[i, predicted_label].item()
                            ranking[true_label].append((img_idx, confidence))
                    # For fake images, include all regardless of correct classification.
                    elif true_label == 1:
                        # Use the confidence for the fake class (index 1)
                        confidence = logits[i, 1].item()
                        ranking[true_label].append((img_idx, confidence))
                    img_idx += 1
                    
        # Sortiere für jede Klasse absteigend nach confidence:
        for cls in ranking:
            ranking[cls] = sorted(ranking[cls], key=lambda x: x[1], reverse=True)
        return ranking

    def get_sorted_indices(self):
        """
        Select indices from the ranking.
        For grid creation, we use the top k images per class.
        For real images (class 0), we need k * (grid_cells - 1) images.
        For fake images (class 1), we need k images.
        """
        k = self.max_grids  # Number of grids requested.
        indices = {}
        for cls in [0, 1]:
            cls_list = self.sorted_confs.get(cls, [])
            if cls == 0:
                # For reals: grid_cells - 1 per grid.
                required = k * (self.grid_size[0] * self.grid_size[1] - 1)
            else:
                # For fakes: 1 per grid.
                required = k
            indices[cls] = [idx for idx, conf in cls_list[:required]]
        return indices

    def get_image_by_index(self, idx):
        """
        Gibt das Bild (als Tensor) anhand eines Indexes aus dem Dataset zurück.
        Voraussetzung: self.dataset ist vorhanden und implementiert __getitem__.
        """
        if self.dataset is None:
            raise ValueError("Kein Dataset übergeben, um Bilder per Index abzurufen.")
        image, _, _, _ = self.dataset[idx]
        return image

    def save_ranking(self, ranking, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(ranking, f)

    def load_ranking(self, file_path):
        with open(file_path, "rb") as f:
            ranking = pickle.load(f)
        return ranking

    def create_GPG_grids(self):
        """
        Create grids using the pre-assessed rankings of both real and fake images.
        For each grid, select one top-ranked fake image and (n_imgs - 1) top-ranked real images.
        These selections are removed from their rankings so that each image is used only once.
        Hier werden bereits vorverarbeitete Tensoren (aus dem DataLoader oder Dateisystem) verwendet.
        """
        print(f"[DEBUG] Creating grids using ranking. Output folder: {self.output_folder}")
        
        # Check if enough grid files already exist.
        existing_files = [f for f in os.listdir(self.output_dir_3ch) if f.endswith('.pt')]
        if len(existing_files) >= self.max_grids:
            print(f"[DEBUG] Found {len(existing_files)} existing grid files in {self.output_dir_3ch}. Skipping grid creation.")
            return

        print(f"[DEBUG] Creating grids using sorted indices. Output folder: {self.output_folder}")
        sorted_indices = self.get_sorted_indices()
        print(f"[DEBUG] Selected sorted indices: {sorted_indices}")
        
        # Define ranked_real and ranked_fake from sorted_indices
        ranked_real = sorted_indices.get(0, [])
        ranked_fake = sorted_indices.get(1, [])

        n_imgs = int(self.grid_size[0]) * int(self.grid_size[1])
        grid_count = 0
        side = int(np.sqrt(n_imgs))  # Assumes a square grid.
        
        # For each grid, select one fake and (n_imgs - 1) reals.
        while grid_count < self.max_grids:
            if len(ranked_fake) < 1:
                print("[DEBUG] Not enough fake images left, stopping grid creation.")
                break
            if len(ranked_real) < (n_imgs - 1):
                print(f"[DEBUG] Not enough real images (need {n_imgs-1}), stopping grid creation.")
                break

            fake_idx = ranked_fake.pop(0)
            fake_img = self.get_image_by_index(fake_idx)
            selected_real_indices = ranked_real[:n_imgs - 1]
            ranked_real = ranked_real[n_imgs - 1:]
            selected_real = [self.get_image_by_index(idx) for idx in selected_real_indices]
            
            images = selected_real + [fake_img]
            random.shuffle(images)
            
            # Find the index of fake_img using torch.equal.
            fake_index = None
            for i, img in enumerate(images):
                if torch.equal(img, fake_img):
                    fake_index = i
                    break
            if fake_index is None:
                raise ValueError("Fake image not found in the grid images list.")
                
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            print(f"[DEBUG] Creating grid {grid_count} using fake image. Original fake index: {fake_index}, Final fake index: {final_fake_index}")
            
            # Staple die Bild-Tensoren (alle sind CPU-Tensoren)
            stacked = torch.stack(images, dim=0)
            grid_tensor = stacked.view(-1, side, side, *stacked.shape[-3:]) \
                                .permute(0, 3, 2, 4, 1, 5) \
                                .reshape(-1,
                                        stacked.shape[1],
                                        stacked.shape[2] * side,
                                        stacked.shape[3] * side)
            
            base_name = f"grid_{grid_count}_fake_{final_fake_index}.pt"
            path_3ch = os.path.join(self.output_dir_3ch, base_name)
            torch.save(grid_tensor, path_3ch)
            print(f"[DEBUG] Saved 3-channel grid tensor to: {path_3ch}")
            
            grid_count += 1

        print("[DEBUG] Grid creation complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    parser.add_argument("--base_output_dir", type=str, default=BASE_OUTPUT_DIR, help="Base output directory for grids.")
    parser.add_argument("--max_grids", type=int, default=MAX_GRIDS, help="Maximum number of grids to create.")
    parser.add_argument("--xai_method", type=str, default=XAI_METHOD, choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model configuration file.")
    parser.add_argument("--grid_split", type=int, default=GRID_SPLIT, help="Grid size for evaluation (e.g. 3 for a 3x3 grid).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    grid_size = (args.grid_split, args.grid_split)
    print(f"[DEBUG] XAI: {args.xai_method}, Base: {args.base_output_dir}, Model: {args.model_path}, Grid: {args.grid_split}x{args.grid_split}")
    
    # Load configuration.
    config = load_config(args.model_path, additional_args=ADDITIONAL_ARGS)
    
    # Load the model.
    model = load_model(config)
    model.eval()
    # For Debugging auf der CPU bleiben wir auf CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[DEBUG] Loaded {model.__class__.__name__} model onto {device}")
    
    # Extrahiere Modell- und Gewichtsnamen für den Ausgabepfad.
    model_name = config.get("model_name", "defaultModel")
    pretrained_path = config['pretrained']
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Gewichtedatei nicht gefunden: {pretrained_path}")
    weights_name = os.path.basename(pretrained_path).split('.')[0]
    
    from train import prepare_testing_data
    
    test_data_loaders = prepare_testing_data(config)
    test_loader = test_data_loaders[list(test_data_loaders.keys())[0]]
    
    # Instanziere den RankedGPGCreator und übergebe zusätzlich den DataLoader.
    grid_creator = RankedGPGCreator(
        base_output_dir=args.base_output_dir,
        grid_size=grid_size,
        max_grids=args.max_grids,
        model=model,
        model_name=model_name,
        weights_name=weights_name,
        loader=test_loader,  # or your test DataLoader
        dataset=test_loader.dataset  # Pass the dataset instance here!
    )
    grid_creator.create_GPG_grids()

    # Optionale Ausgabe: Inspektion der erstellten Grids
    grid_dir = os.path.join(grid_creator.output_folder, "3ch")
    grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]
    print(f"[DEBUG] Found {len(grid_paths)} grid tensors for evaluation in {grid_dir}.")

    # Lade die Grid-Tensoren.
    preprocessed_tensors = []
    for grid_path in grid_paths:
        grid_tensor = torch.load(grid_path, map_location=device)
        print(f"[DEBUG] Loaded grid tensor from {grid_path} with shape: {grid_tensor.shape}")
        preprocessed_tensors.append(grid_tensor)

    # Instanziiere den Evaluator unter Verwendung des bereits geladenen Modells.
    if args.xai_method == "bcos":
        evaluator = BCOSEvaluator(config, model=model)  # Adjust as needed.
    elif args.xai_method == "lime":
        evaluator = LIMEEvaluator(config, model=model, device=device)
    elif args.xai_method == "gradcam":
        # evaluator = GradCamEvaluator(config, model=model)  # Uncomment if implemented.
        print("[DEBUG] GradCAM evaluator not implemented in this example.")
        return
    else:
        raise ValueError(f"Unknown xai_method: {args.xai_method}")

    # Führe die Evaluation durch.
    evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=args.grid_split)

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py