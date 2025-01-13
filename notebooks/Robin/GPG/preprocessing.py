import torch
from torchvision import transforms
from PIL import Image

# Custom Transformations
class MyToTensor(transforms.ToTensor):
    def __call__(self, input_img):
        if not isinstance(input_img, torch.Tensor):
            return super().__call__(input_img)
        return input_img

class AddInverse(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        return torch.cat([in_tensor, 1 - in_tensor], self.dim)

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to (224, 224)
    MyToTensor(),                   # Convert to tensor
    AddInverse(dim=0),              # Add inverse channels
])

def preprocess_image(image_path):
    """
    Preprocess an image by resizing and adding inverse channels.

    Args:
        image_path (str): Path to the image.

    Returns:
        torch.Tensor: Preprocessed image.
    """
    img = Image.open(image_path).convert("RGB")  # Load image and convert to RGB
    return transform_pipeline(img)
    