
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, XGradCAM, GradCAMPlusPlus
from captum.attr import LayerGradCam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Auto-find the last Conv2d layer that isn't "adjust" or "proj"
def find_last_valid_conv_layer(module):
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and all(x not in name for x in ["adjust", "proj"]):
            last_conv = m
    return last_conv

class GradCamEvaluator:
    def __init__(self, model, method="gradcam"):
        self.model = model
        self.method = method.lower()
        self.target_layer = find_last_valid_conv_layer(self.model.backbone)

        if self.target_layer is None:
            raise ValueError("No valid Conv2d layer found in model backbone.")

        if self.method == "gradcam":
            self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        elif self.method == "xgrad":
            self.cam = XGradCAM(model=self.model, target_layers=[self.target_layer])
        elif self.method == "grad++":
            self.cam = GradCAMPlusPlus(model=self.model, target_layers=[self.target_layer])
        elif self.method == "layergrad":
            self.cam = LayerGradCam(model=self.model, layer=self.target_layer)
        else:
            raise ValueError(f"Unknown CAM method: {self.method}")

    def evaluate(self, input_tensor, label):
        if self.method == "layergrad":
            attributions = self.cam.attribute(input_tensor, target=label)
            return attributions.squeeze().cpu().detach().numpy()
        else:
            targets = [ClassifierOutputTarget(label)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0]
