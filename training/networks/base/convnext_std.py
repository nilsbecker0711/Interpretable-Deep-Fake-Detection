'''
Standard (non-b-cos) ConvNeXt backbone — the plain twin of convnext_bcos.

Wraps torchvision's ConvNeXt the same way resnet34.py wraps torchvision's
ResNet. convnext_tiny at num_classes=2 is 27.82M params, matching the b-cos
convnext_bcos "tiny" at 27.78M, so the pair is size-matched like the
resnet34 / resnet34_bcos_v2 and xception / xception_bcos pairs.

NOTE: the other two files named convnext in this repo
(networks/base/convnext.py and detectors/convnext.py) are both copies of the
*b-cos* ConvNeXt, not a standard one — neither is wired into the registry.
'''

import logging

import torch
import torch.nn as nn
import torchvision

from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

_VARIANTS = {
    "tiny": torchvision.models.convnext_tiny,
    "small": torchvision.models.convnext_small,
    "base": torchvision.models.convnext_base,
    "large": torchvision.models.convnext_large,
}


@BACKBONE.register_module(module_name="convnext")
class ConvNeXt(nn.Module):
    def __init__(self, convnext_config):
        super(ConvNeXt, self).__init__()
        """ Constructor
        Args:
            convnext_config: configuration file with the dict format
        """
        self.num_classes = convnext_config["num_classes"]
        inc = convnext_config.get("inc", 3)
        # 'atto' exists only in the b-cos twin (it is not a torchvision variant).
        variant = convnext_config.get("block_setting", "tiny")
        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown ConvNeXt variant {variant!r}; expected one of "
                f"{sorted(_VARIANTS)}")

        convnext = _VARIANTS[variant](weights=None, num_classes=self.num_classes)
        self.convnext = convnext.features
        self.avgpool = convnext.avgpool
        # torchvision's head is Sequential(LayerNorm2d, Flatten, Linear); it is
        # kept whole so the norm stays paired with its linear layer.
        self.head = convnext.classifier

    def features(self, inp):
        x = self.convnext(inp)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = self.head(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out

    def get_gradcam_target(self) -> nn.Module:
        """Grad-CAM target: the last feature map, i.e. the tensor entering global
        pooling. Declared rather than discovered so it cannot drift onto the
        LayerNorm2d inside the head (see convnext_bcos / xception_bcos, where the
        shape heuristic picks the head's input norm).
        """
        return self.convnext

    def initialize_weights(self, module):
        # In line with the b-cos v2 initialization: only the weight-carrying
        # leaves are initialized, everything else keeps its default.
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
