import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Custom transformation class that safely converts images to tensors
class MyToTensor(transforms.ToTensor):
    def __call__(self, input_img):
        if not isinstance(input_img, torch.Tensor):
            return super().__call__(input_img)
        return input_img

# Dataset loading images from folders
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_paths, transform=None):
        self.image_files = []
        for fp, label in folder_paths.items():
            for root, _, files in os.walk(fp):
                for f in files:
                    if f.endswith((".png", ".jpg")):
                        self.image_files.append((os.path.join(root, f), label))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, label = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    MyToTensor(),
])

# Point to your folder
file_path_deepfakebench = {
    "/Users/msrobin/GitHub Repositorys/Interpretable-Deep-Fake-Detection-2/datasets/2x2_images": 1
}

# Load dataset and dataloader
dataset = CustomImageDataset(file_path_deepfakebench, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load your model
from training.detectors import DETECTOR
import yaml

def load_config(path, additional_args={}):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config.update(additional_args)
    return config

path = "./training/config/detector/xception.yaml"
additional_args = {'test_batchSize': 12, 'pretrained': './weights/ckpt_best.pth'}
config = load_config(path, additional_args=additional_args)
model_class = DETECTOR[config['model_name']]
model = model_class(config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define last conv layer
TARGET_LAYER = model.backbone.fea_part5
cam = XGradCAM(model=model, target_layers=[TARGET_LAYER])

# Grad-CAM Visualization
for img_batch, label_batch, path_batch in dataloader:
    print(f"Batch of images shape: {img_batch.shape}")

    for i in range(min(8, len(img_batch))):
        img = img_batch[i].unsqueeze(0).to(device)
        label = label_batch[i]
        img_path = path_batch[i]
        img.requires_grad = True

        # Forward pass
        data_dict = {'image': img, 'label': label}
        model.zero_grad()
        pred_dict = model(data_dict)
        prob = pred_dict['prob'].item()
        pred_class = int(prob > 0.5)

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=img, targets=targets)[0]

        # Prepare overlay
        img_np = np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_np)
        ax[0].axis('off')
        ax[0].set_title("Original Image")

        label_text = "Fake" if pred_class == 1 else "Real"
        ax[1].imshow(overlay)
        ax[1].axis('off')
        ax[1].set_title(f"Grad-CAM: {label_text} ({prob*100:.2f}%)")

        plt.show()
