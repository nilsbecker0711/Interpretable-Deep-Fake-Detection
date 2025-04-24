'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for ResNet34 backbone.
'''

import os
import logging
from typing import Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="resnet34")
class ResNet34(nn.Module):
    def __init__(self, resnet_config):
        super(ResNet34, self).__init__()
        """ Constructor
        Args:
            resnet_config: configuration file with the dict format
        """
        self.num_classes = resnet_config["num_classes"]
        inc = resnet_config["inc"]
        self.mode = resnet_config["mode"]

        # Define layers of the backbone
        resnet = torchvision.models.resnet34(pretrained=False)  # FIXME: download the pretrained weights from online
        # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )


    def features(self, inp):
        x = self.resnet(inp)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)# or nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Recursively apply to custom modules
        elif isinstance(module, (torchvision.models.resnet.BasicBlock)):
            for submodule in module.children():
                self.initialize_weights(submodule)
        # Ignore activation, pooling, and sequential layers
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            pass  # Do nothing
        else:
            print(f'unknown module type {type(module)}')