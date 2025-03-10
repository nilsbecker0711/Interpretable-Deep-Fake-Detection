import os
import sys
import pickle
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import yaml

sys.path.append("/Users/Linus/Desktop/GIThubXAIFDEEPFAKE/Interpretable-Deep-Fake-Detection")
from training.detectors.xception_detector import XceptionDetector
from training.detectors import DETECTOR

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
        with open('./training/config/train_config.yaml', 'r') as f:
            config2 = yaml.safe_load(f)
    except FileNotFoundError:
        with open(os.path.expanduser('~/Interpretable-Deep-Fake-Detection/training/config/train_config.yaml'), 'r') as f:
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

class RankedGPGCreator:
    def __init__(self, real_dir, fake_dir, base_output_dir, grid_size=(3, 3), max_grids=2,
                 model=None, model_name="default", weights_name="default"):
        """
        Args:
            real_dir (str): Directory with real images.
            fake_dir (str): Directory with fake images.
            base_output_dir (str): Base directory in which to save grids.
            grid_size (tuple): Dimensions of the grid (rows, columns).
            max_grids (int): Maximum number of grids to create.
            model (callable): A model that takes an image tensor and returns a confidence score.
            model_name (str): Name of the model (used for naming the folder).
            weights_name (str): Name of the weights (used for naming the folder).
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.grid_size = grid_size
        self.max_grids = max_grids
        self.model = model

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
        Only include images that are correctly classified as real (ground-truth label 0)
        if available; otherwise, include misclassified images sorted by confidence (lowest first).
        Returns:
            List of real image paths sorted by confidence.
            (Correct predictions sorted descending by confidence; if none, misclassified ones sorted ascending.)
        """
        real_images = self.get_all_png_files(self.real_dir)
        print(f"[DEBUG] Found {len(real_images)} real images for pre-assessment.")
        transform = T.ToTensor()
        
        correct_confidences = []   # For images predicted as real (label 0)
        incorrect_confidences = [] # For images predicted incorrectly (not 0)

        for img_path in real_images:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor_img = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]
                # No conversion to 6 channels: we keep the original 3 channels.

            with torch.no_grad():
                true_label = 0  # ground-truth for real images
                data_dict = {
                    'image': tensor_img,
                    'label': torch.tensor([true_label]).to(tensor_img.device)
                }
                output = self.model(data_dict)
                # Assume the model returns logits under the key 'cls'
                logits = output['cls']
                predicted_label = logits.argmax(dim=1).item()
                # Use the logit of the predicted class as confidence.
                confidence = logits[0, predicted_label].item()

            if predicted_label == true_label:
                correct_confidences.append((img_path, confidence))
            else:
                incorrect_confidences.append((img_path, confidence))
                print(f"[DEBUG] Including misclassified real image {img_path}: predicted {predicted_label} != {true_label} with confidence {confidence}")

        # Decide which set to use:
        if correct_confidences:
            # If any images were correctly classified, sort them descending by confidence.
            correct_confidences.sort(key=lambda x: x[1], reverse=True)
            ranked = [img_path for img_path, conf in correct_confidences]
        else:
            # Otherwise, sort the misclassified ones ascending (lowest confidence first).
            incorrect_confidences.sort(key=lambda x: x[1])
            ranked = [img_path for img_path, conf in incorrect_confidences]

        return ranked


    def pre_assess_fake_images(self):
        """
        Pre-assess fake images using the provided model.
        Only include images that are correctly classified as fake (ground-truth label 1),
        similar to LocalisationAnalyser.
        Returns:
            List of fake image paths sorted by confidence (highest first).
        """
        fake_images = self.get_all_png_files(self.fake_dir)
        print(f"[DEBUG] Found {len(fake_images)} fake images for pre-assessment.")
        transform = T.ToTensor()
        image_confidences = []

        for img_path in fake_images:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor_img = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]
                # No concatenation to 6 channels here either.

            with torch.no_grad():
                true_label = 1  # ground-truth label for fake images
                data_dict = {
                    'image': tensor_img,
                    'label': torch.tensor([true_label]).to(tensor_img.device)
                }
                output = self.model(data_dict)
                logits = output['cls']
                predicted_class = logits.argmax(dim=1).item()
                if predicted_class != true_label:
                    print(f"[DEBUG] Skipping fake image {img_path}: predicted {predicted_class} != {true_label}")
                    continue
                confidence = logits[0, predicted_class].item()

            image_confidences.append((img_path, confidence))
        
        image_confidences.sort(key=lambda x: x[1], reverse=True)
        ranked = [img_path for img_path, conf in image_confidences]
        return ranked

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
        """
        print(f"[DEBUG] Creating grids using ranking. real_dir={self.real_dir}, fake_dir={self.fake_dir}, output_folder={self.output_folder}")
        

        # Check if enough grid files already exist.
        existing_files = [f for f in os.listdir(self.output_dir_3ch) if f.endswith('.pt')]
        if len(existing_files) >= self.max_grids:
            print(f"[DEBUG] Found {len(existing_files)} existing grid files in {self.output_dir_3ch}. Skipping grid creation.")
            return

        n_imgs = int(self.grid_size[0]) * int(self.grid_size[1])
        grid_count = 0
        side = int(np.sqrt(n_imgs))  # Assumes a square grid.
        
        # Use copies of the pre-assessed rankings so that original lists remain intact if needed.
        ranked_real_images = self.real_images_ranked.copy()
        ranked_fake_images = self.fake_images_ranked.copy()
        
        # Continue creating grids while we have enough images.
        while grid_count < self.max_grids:
            if len(ranked_fake_images) < 1:
                print("[DEBUG] Not enough fake images left, stopping grid creation.")
                break
            if len(ranked_real_images) < (n_imgs - 1):
                print(f"[DEBUG] Not enough ranked real images (need {n_imgs-1}), stopping grid creation.")
                break
            
            # Select one fake image and remove it from the ranking.
            fake_img_path = ranked_fake_images.pop(0)
            # Select the top (n_imgs - 1) real images and remove them from the ranking.
            selected_real = ranked_real_images[:n_imgs - 1]
            ranked_real_images = ranked_real_images[n_imgs - 1:]
            
            # Combine the selected real images with the fake image.
            images = selected_real + [fake_img_path]
            random.shuffle(images)
            
            fake_index = images.index(fake_img_path)
            final_fake_index = (fake_index % side) * side + (fake_index // side)
            print(f"[DEBUG] Creating grid {grid_count} using fake image: {fake_img_path}")
            print(f"[DEBUG] Original fake index: {fake_index}, Final fake index: {final_fake_index}")
            
            transform = T.ToTensor()
            png_tensors = []
            for img_path in images:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    png_tensors.append(transform(img))
            print(f"[DEBUG] Number of loaded PNG tensors: {len(png_tensors)}")
            
            # Stack the image tensors.
            stacked = torch.stack(png_tensors, dim=0)
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