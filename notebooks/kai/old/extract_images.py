import tarfile
import io
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class TarImageDataset(Dataset):
    def __init__(self, tar_path, transform=None):
        self.tar_paths = tar_path if isinstance(tar_path, list) else [tar_path]
        self.transform = transform
        self.members = []
        self.label_counts = Counter()  # Counter to store label distribution

        # Collect members from all tar files
        for tar_index, tar_path in enumerate(self.tar_paths):
            self._collect_members(tar_path, tar_index)

    def _collect_members(self, tar_path, tar_index):
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.png'):
                    # Assign label based on directory
                    label = 1 if '/real/' in member.name else 0 if '/fake/' in member.name else None
                    if label is not None:
                        self.members.append((tar_index, member, label))
                        self.label_counts[label] += 1  # Update label count

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        tar_index, member, label = self.members[index]
        
        # Read the image from the specific tar file
        with tarfile.open(self.tar_paths[tar_index], 'r') as tar:
            img_file = tar.extractfile(member)
            if img_file is None:
                raise FileNotFoundError(f"Cannot extract file {member.name} from {self.tar_paths[tar_index]}.")
            image = Image.open(io.BytesIO(img_file.read())).convert('RGB')  # Ensure RGB format
            
            if self.transform:
                image = self.transform(image)  # Apply any transformations
            else:
                # Convert image to tensor if no transform is applied
                image = torch.tensor(np.array(image)).permute(2, 0, 1)  # Change to (C, H, W)
                image = image.float() / 255.0  # Normalize to [0, 1]
        return image, label
    
    def _preprocess_image(self, image):
        """Preprocess the image to a model-readable format."""
        # Convert the image to a tensor and apply normalization if needed
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image)).permute(2, 0, 1)  # Change from HWC to CHW
            image = image.float() / 255.0  # Normalize to [0, 1]
        return image
    
    def get_label_distribution(self):
        return dict(self.label_counts)  # Return the distribution as a dictionary