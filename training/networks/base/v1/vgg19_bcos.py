import os
import logging
from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import BACKBONE
from bcos.bcosconv2d import BcosConv2d
from bcos.utils import MyAdaptiveAvgPool2d, FinalLayer
import math

logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> BcosConv2d:
    """3x3 convolution with padding"""
    #print(in_planes)
    scale = 10000/(math.sqrt(in_planes*9)) # 1000/root(kernel^2*channels)
    #scale = 10000
    return BcosConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, b = 1.25, scale_fact=scale)


class VGG19_Bcos(nn.Module):
    def __init__(self, vgg_config):
        
        super(VGG19_Bcos, self).__init__()
        """ Constructor
        Args:
            vgg_config: configuration file with the dict format
        """
        self.mode = vgg_config["mode"]
        self.num_classes = vgg_config["num_classes"]
        self.bias = vgg_config["bias"]
        self.temperature = vgg_config["temperature"]
        self.b = vgg_config["b"]
        self.config = vgg_config

        self.features = self._make_layers()
        #self.classifier = BcosConv2d(512, self.num_classes, kernel_size=1, max_out=2)
        #MyAdaptiveAvgPool2d((1, 1)),
        #FinalLayer(bias=self.bias, norm=self.temperature)
        self.classifier = nn.Sequential(
            BcosConv2d(512, self.num_classes, kernel_size=1, max_out=2, b = self.b, scale_fact=1000/math.sqrt(512*9)),
            MyAdaptiveAvgPool2d((1, 1)),
            FinalLayer(bias=self.bias, norm=self.temperature)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layers(self):
        layers = []
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
               512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        in_channels = self.config["input_channels"]  # Default input channel size

        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(conv3x3(in_channels, x))
                in_channels = x
        
        return nn.Sequential(*layers)

    def _vgg_impl(self, x):
        """ Forward implementation for feature extraction """
        x = self.features(x)
        print(x.shape)
        return x

    def classifier_head(self, features):
        print(features.shape)
        x = self.classifier(features)
        if self.num_classes == 1:
            x = x.squeeze()
        return x

    def classifier_head2(self, features):
        x = self.classifier(features)  # [batch_size, num_classes, 7, 7]
        (print(f'in classifier head {x}'))
        x = F.adaptive_avg_pool2d(x, 1)  # Pool to [batch_size, num_classes, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # Remove last two dimensions -> [batch_size, num_classes]
        return x

    def forward(self, inp):
        x = self._vgg_impl(inp)
        # Adds spatial dimensions for BCosConv2d compatibility
        out = self.classifier_head2(x)
        return out

@BACKBONE.register_module(module_name="vgg19_bcos")
def build_vgg19_bcos(vgg_config):
    print("vgg init")
    return VGG19_Bcos(vgg_config)