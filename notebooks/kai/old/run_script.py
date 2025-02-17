import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from b_cos.resnet import resnet50
from b_cos.bcosconv2d import BcosConv2d
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MyToTensor(transforms.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_img):
        if not isinstance(input_img, torch.Tensor):
            return super().__call__(input_img)
        return input_img

class AddInverse(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        return torch.cat([in_tensor, 1-in_tensor], self.dim)

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_paths, transform=None):
        self.image_files = []
        for fp, label in folder_paths.items():
            subfolders = [os.path.join(fp, d) for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
            for folder_path in subfolders:
                self.image_files.extend(
                    [(os.path.join(folder_path, f), label) for f in os.listdir(folder_path) if f.endswith((".png", ".jpg"))]
                )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, label = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def load_pretrained_weights(model, model_url):
    state_dict = load_state_dict_from_url(model_url, progress=True)
    adapted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("conv", "conv.linear").replace("fc", "fc.linear")
        if new_key in model.state_dict() and model.state_dict()[new_key].shape == value.shape:
            adapted_state_dict[new_key] = value
    model.load_state_dict(adapted_state_dict, strict=False)
    nn.init.kaiming_normal_(model.fc.linear.weight)
    if model.fc.linear.bias is not None:
        model.fc.linear.bias.data.zero_()

def train_model(dataset_path, num_epochs=1, batch_size=32, learning_rate=0.001, state_dict_path="None"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        MyToTensor(),
        AddInverse(dim=0),
    ])

    # Define file paths for the dataset
    file_path_deepfakebench = {
        os.path.join(dataset_path, 'manipulated_sequences/Deepfakes/c40/frames'): 1,
        os.path.join(dataset_path, 'manipulated_sequences/Face2Face/c40/frames'): 1,
        os.path.join(dataset_path, 'original_sequences/actors/c40/frames'): 0,
        os.path.join(dataset_path, 'original_sequences/youtube/c40/frames'): 0,
    }

    dataset = CustomImageDataset(file_path_deepfakebench, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = resnet50(pretrained=False, progress=True, num_classes=1, groups=32, width_per_group=4)
    if state_dict_path == "None":
        load_pretrained_weights(model, 'https://download.pytorch.org/models/resnet50-19c8e357.pth')
    else:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.float()
            if images.shape[0] == 1: # IMPORTANT for handling a batch size of 1 correctly!!!
                outputs = outputs.view(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
        timestamp = time.time()
        model_filename = f'b_cos_model_{timestamp:.2f}.pth'
        torch.save(model.state_dict(), model_filename)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model with a dataset")
    # parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--state-dict-path', type=str, default="None", help='Path to saved state_dictionary')
    args = parser.parse_args()
    # dataset_path = '../../DeepfakeBench/datasets/rgb/FaceForensics++/'
    # dataset_path = '/home/ma/ma_ma/ma_kreffert/interpretable-deep-fake-detection/DeepfakeBench/datasets/rgb/FaceForensics++/'
    dataset_path = '/pfs/work7/workspace/scratch/ma_tischuet-team-project-hws-2024/datasets/rgb/FaceForensics++/'
    
    train_model(dataset_path, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                state_dict_path=args.state_dict_path)
