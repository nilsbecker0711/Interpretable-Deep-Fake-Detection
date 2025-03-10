#!/usr/bin/env python
import os
import sys
import torch
import argparse
import numpy as np
import yaml
import pickle
import random
import torchvision.transforms as T
from PIL import Image

# Evaluators.
from B_COS_eval import BCOSEvaluator
from LIME_eval import LIMEEvaluator  # Adjust the import path if needed.
# from GRADCAM_eval import GradCamEvaluator  # Uncomment if available.

PROJECT_PATH = "/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection"
sys.path.append(PROJECT_PATH)
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR


XAI_METHOD = "lime"
BASE_OUTPUT_DIR = "datasets/GPG_grids"
GRID_SPLIT = 3
REAL_DIR = "datasets/FaceForensics++/original_sequences/actors/c40/frames"
FAKE_DIR = "datasets/FaceForensics++/manipulated_sequences/DeepFakeDetection/c40/frames"
MAX_GRIDS = 2
MODEL_PATH = os.path.join(PROJECT_PATH, "training/config/detector/xception.yaml")

ADDITIONAL_ARGS = {
    "model_name": "xception",
    "test_batchSize": 12,
    "pretrained": os.path.join(PROJECT_PATH, "weights/resnet/ckpt_best.pth")
}

def load_model(config):
    """
    Load and return the model using the DETECTOR registry.
    """
    print("Registered models:", DETECTOR.data.keys())
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
    def __init__(self, real_dir, fake_dir, base_output_dir, grid_size=(3, 3), max_grids=2,
                 model=None, model_name="default", weights_name="default",
                 real_loader=None, fake_loader=None):
        """
        Args:
            real_dir (str): Directory with real images (wird nur genutzt, wenn kein Loader übergeben wird).
            fake_dir (str): Directory with fake images.
            base_output_dir (str): Base directory in which to save grids.
            grid_size (tuple): Dimensions of the grid (rows, columns).
            max_grids (int): Maximum number of grids to create.
            model (callable): A model that takes an image tensor and returns a confidence score.
            model_name (str): Name of the model (used for naming the folder).
            weights_name (str): Name of the weights (used for naming the folder).
            real_loader (DataLoader, optional): Falls übergeben, werden reale Bilder aus dem DataLoader geladen.
            fake_loader (DataLoader, optional): Falls übergeben, werden fake Bilder aus dem DataLoader geladen.
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.grid_size = grid_size
        self.max_grids = max_grids
        self.model = model
        self.real_loader = real_loader
        self.fake_loader = fake_loader

        # Name output folder after model name and weights.
        self.model_name = model_name
        self.weights_name = weights_name
        self.output_folder = os.path.join(base_output_dir, f"{model_name}_{weights_name}")
        
        # Create subdirectories for 3-channel and 6-channel grids.
        self.output_dir_3ch = os.path.join(self.output_folder, "3ch")
        self.output_dir_6ch = os.path.join(self.output_folder, "6ch")
        os.makedirs(self.output_dir_3ch, exist_ok=True)
        os.makedirs(self.output_dir_6ch, exist_ok=True)

        # Process real images ranking.
        self.ranking_file = os.path.join(self.output_folder, "ranking.pkl")
        if os.path.exists(self.ranking_file):
            self.real_images_ranked = self.load_ranking(self.ranking_file)
            print(f"[DEBUG] Loaded existing real ranking from {self.ranking_file}")
        else:
            self.real_images_ranked = self.pre_assess_images()
            self.save_ranking(self.real_images_ranked, self.ranking_file)
            print(f"[DEBUG] Saved real ranking to {self.ranking_file}")

        # Process fake images ranking.
        self.fake_ranking_file = os.path.join(self.output_folder, "fake_ranking.pkl")
        if os.path.exists(self.fake_ranking_file):
            self.fake_images_ranked = self.load_ranking(self.fake_ranking_file)
            print(f"[DEBUG] Loaded existing fake ranking from {self.fake_ranking_file}")
        else:
            self.fake_images_ranked = self.pre_assess_fake_images()
            self.save_ranking(self.fake_images_ranked, self.fake_ranking_file)
            print(f"[DEBUG] Saved fake ranking to {self.fake_ranking_file}")

    def get_all_png_files(self, root_folder, filter_keyword=None):
        """Recursively collect all .png file paths from a given root folder."""
        png_files = []
        for dirpath, _, filenames in os.walk(root_folder):
            if filter_keyword and filter_keyword not in dirpath:
                continue
            for file in filenames:
                if file.endswith('.png'):
                    png_files.append(os.path.join(dirpath, file))
        return png_files

    def pre_assess_images(self):
        """
        Pre-assess real images using the provided model.
        Nutzt entweder den DataLoader (falls vorhanden) oder die Dateisuche.
        """
        if self.real_loader is not None:
            real_images = get_images_from_dataloader(self.real_loader)
            print(f"[DEBUG] Loaded {len(real_images)} real images from DataLoader.")
        else:
            real_paths = self.get_all_png_files(self.real_dir)
            print(f"[DEBUG] Found {len(real_paths)} real images for pre-assessment.")
            transform = T.ToTensor()
            real_images = []
            for path in real_paths:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    real_images.append(transform(img))
        
        correct_confidences = []   # For images predicted as real (label 0)
        incorrect_confidences = [] # For images predicted incorrectly (not 0)

        for img_tensor in real_images:
            img_tensor = img_tensor.unsqueeze(0)  # shape: [1, 3, H, W]
            with torch.no_grad():
                true_label = 0  # ground-truth for real images
                data_dict = {
                    'image': img_tensor,
                    'label': torch.tensor([true_label])  # remains on CPU
                }
                output = self.model(data_dict)
                logits = output['cls']
                predicted_label = logits.argmax(dim=1).item()
                confidence = logits[0, predicted_label].item()

            if predicted_label == true_label:
                correct_confidences.append((img_tensor, confidence))
            else:
                incorrect_confidences.append((img_tensor, confidence))
                print(f"[DEBUG] Including misclassified real image with confidence {confidence}")

        if correct_confidences:
            correct_confidences.sort(key=lambda x: x[1], reverse=True)
            ranked = [tensor for tensor, conf in correct_confidences]
        else:
            incorrect_confidences.sort(key=lambda x: x[1])
            ranked = [tensor for tensor, conf in incorrect_confidences]

        return ranked

    def pre_assess_fake_images(self):
        """
        Pre-assess fake images using the provided model.
        Nutzt entweder den DataLoader (falls vorhanden) oder die Dateisuche.
        """
        if self.fake_loader is not None:
            fake_images = get_images_from_dataloader(self.fake_loader)
            print(f"[DEBUG] Loaded {len(fake_images)} fake images from DataLoader.")
        else:
            fake_paths = self.get_all_png_files(self.fake_dir)
            print(f"[DEBUG] Found {len(fake_paths)} fake images for pre-assessment.")
            transform = T.ToTensor()
            fake_images = []
            for path in fake_paths:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    fake_images.append(transform(img))
        
        image_confidences = []

        for img_tensor in fake_images:
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                true_label = 1  # ground-truth for fake images
                data_dict = {
                    'image': img_tensor,
                    'label': torch.tensor([true_label])
                }
                output = self.model(data_dict)
                logits = output['cls']
                predicted_class = logits.argmax(dim=1).item()
                if predicted_class != true_label:
                    print(f"[DEBUG] Skipping fake image: predicted {predicted_class} != {true_label}")
                    continue
                confidence = logits[0, predicted_class].item()
            image_confidences.append((img_tensor, confidence))
        
        image_confidences.sort(key=lambda x: x[1], reverse=True)
        ranked = [tensor for tensor, conf in image_confidences]
        return ranked

    def save_ranking(self, ranking, file_path):
        # Da wir nun Tensoren (anstatt Dateipfade) ranken, speichern wir diese mittels pickle.
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

        n_imgs = int(self.grid_size[0]) * int(self.grid_size[1])
        grid_count = 0
        side = int(np.sqrt(n_imgs))  # Assumes a square grid.
        
        # Kopien der Rankings, damit das Original erhalten bleibt.
        ranked_real = self.real_images_ranked.copy()
        ranked_fake = self.fake_images_ranked.copy()
        
        while grid_count < self.max_grids:
            if len(ranked_fake) < 1:
                print("[DEBUG] Not enough fake images left, stopping grid creation.")
                break
            if len(ranked_real) < (n_imgs - 1):
                print(f"[DEBUG] Not enough ranked real images (need {n_imgs-1}), stopping grid creation.")
                break
            
            fake_img = ranked_fake.pop(0)
            selected_real = ranked_real[:n_imgs - 1]
            ranked_real = ranked_real[n_imgs - 1:]
            
            images = selected_real + [fake_img]
            random.shuffle(images)
            
            # Bestimme den Index des Fake-Bildes im Grid
            fake_index = images.index(fake_img)
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
            
            six_ch_tensor = torch.cat([grid_tensor, 1.0 - grid_tensor], dim=1)
            path_6ch = os.path.join(self.output_dir_6ch, base_name)
            torch.save(six_ch_tensor, path_6ch)
            print(f"[DEBUG] Saved 6-channel grid tensor to: {path_6ch}")
            
            grid_count += 1

        print("[DEBUG] Grid creation complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate grids using a specified model and XAI method.")
    parser.add_argument("--real_dir", type=str, default=REAL_DIR, help="Directory with real images.")
    parser.add_argument("--fake_dir", type=str, default=FAKE_DIR, help="Directory with fake images.")
    parser.add_argument("--base_output_dir", type=str, default=BASE_OUTPUT_DIR, help="Base output directory for grids.")
    parser.add_argument("--max_grids", type=int, default=MAX_GRIDS, help="Maximum number of grids to create.")
    parser.add_argument("--xai_method", type=str, default=XAI_METHOD, choices=["bcos", "lime", "gradcam"], help="XAI method to use.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model configuration file.")
    parser.add_argument("--grid_split", type=int, default=GRID_SPLIT, help="Grid size for evaluation (e.g. 3 for a 3x3 grid).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    grid_size = (args.grid_split, args.grid_split)
    print(f"XAI: {args.xai_method}, Base: {args.base_output_dir}, Model: {args.model_path}, Grid: {args.grid_split}x{args.grid_split}")
    print(f"Real: {args.real_dir}, Fake: {args.fake_dir}, Max grids: {args.max_grids}")
    
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
    pretrained_path = config.get("pretrained", "default_weights.pth")
    weights_name = os.path.basename(pretrained_path).split('.')[0]
    
    from train import prepare_testing_data
    test_data_loaders = prepare_testing_data(config)
    test_loader = test_data_loaders[list(test_data_loaders.keys())[0]]
    
    # Instanziere den RankedGPGCreator und übergebe zusätzlich den DataLoader.
    grid_creator = RankedGPGCreator(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        base_output_dir=args.base_output_dir,
        grid_size=grid_size,
        max_grids=args.max_grids,
        model=model,
        model_name=model_name,
        weights_name=weights_name,
        real_loader=test_loader,   # Nutze den DataLoader für reale Bilder
        fake_loader=test_loader    # Falls nötig, auch für Fake-Bilder (oder separaten Loader)
    )
    grid_creator.create_GPG_grids()

    # Optionale Ausgabe: Inspektion der erstellten Grids
    grid_dir = os.path.join(grid_creator.output_folder, "6ch") if args.xai_method == "bcos" else os.path.join(grid_creator.output_folder, "3ch")
    grid_paths = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.pt')]
    print(f"Found {len(grid_paths)} grid tensors for evaluation in {grid_dir}.")

    # Lade die Grid-Tensoren.
    preprocessed_tensors = []
    for grid_path in grid_paths:
        grid_tensor = torch.load(grid_path, map_location="cpu")
        print(f"[DEBUG] Loaded grid tensor from {grid_path} with shape: {grid_tensor.shape}")
        preprocessed_tensors.append(grid_tensor)

    # Instanziiere den Evaluator unter Verwendung des bereits geladenen Modells.
    if args.xai_method == "bcos":
        evaluator = BCOSEvaluator(config, model=model)  # Adjust as needed.
    elif args.xai_method == "lime":
        evaluator = LIMEEvaluator(config, model=model)
    elif args.xai_method == "gradcam":
        # evaluator = GradCamEvaluator(config, model=model)  # Uncomment if implemented.
        print("GradCAM evaluator not implemented in this example.")
        return
    else:
        raise ValueError(f"Unknown xai_method: {args.xai_method}")

    # Führe die Evaluation durch.
    evaluator.evaluate(preprocessed_tensors, grid_paths, grid_split=args.grid_split)

if __name__ == "__main__":
    main()

# python notebooks/Linus/GridPointingGame/GPG_eval.py