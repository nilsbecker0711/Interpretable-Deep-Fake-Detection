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


from bcos.bcosconv2d import BcosConv2d
from bcos.detector_utils import MyAdaptiveAvgPool2d, FinalLayer
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# from data.data_transforms import AddInverse
# from .bcosconv2d import BcosConv2d
from torch.hub import load_state_dict_from_url
# from torch.utils.model_zoo import load_url as load_state_dict_from_url
# from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional



logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, b: int=2) -> BcosConv2d:
    """3x3 convolution with padding"""
    return BcosConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, b=b)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, b: int=2) -> BcosConv2d:
    """1x1 convolution"""
    return BcosConv2d(in_planes, out_planes, kernel_size=1, stride=stride, b=b)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        short_cat=False,
        b: int=2
    ) -> None:
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, b=b)
        self.conv2 = conv3x3(planes, planes, b=b)
        self.downsample = downsample
        self.stride = stride
        self.short_cat = BcosConv2d(2 * planes, planes, b=b) if short_cat else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.short_cat is not None:
            out = self.short_cat(torch.cat([out, identity], dim=1))
        else:
            out += identity

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        short_cat = False, 
        b: int=2
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, b=b)
        self.conv2 = conv3x3(width, width, stride, b=b)
        self.conv3 = conv1x1(width, planes * self.expansion, b=b)
        self.downsample = downsample
        self.stride = stride
        self.short_cat = BcosConv2d(2 * planes * self.expansion, planes * self.expansion, b=b) if short_cat else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.short_cat is not None:
            out = self.short_cat(torch.cat([out, identity], dim=1))
        else:
            out += identity

        return out

@BACKBONE.register_module(module_name="resnet34_bcos")
class ResNet34_bcos(nn.Module):

    def __init__(self, resnet_config):
        super(ResNet34_bcos, self).__init__()
        """ Constructor
        Args:
            resnet_config: configuration file with the dict format
        """
        # inc = resnet_config["inc"]
        self.mode = resnet_config["mode"]

        # # Define layers of the backbone
        # resnet = torchvision.models.resnet34(pretrained=True)  # FIXME: download the pretrained weights from online
        # # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, self.num_classes)

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(512, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )


        self.num_classes = resnet_config["num_classes"]
        # resnet34 parameters


        block = BasicBlock
        layers = [3, 4, 6, 3]
        #pretrained, progress,

        # if resnet_config["norm_layer"] is None:
        #     norm_layer = nn.BatchNorm2d
        # self._norm_layer = norm_layer
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.log_temperature: int = resnet_config["log_temperature"]
        self.bias: float = np.log(resnet_config["bias"][0]/resnet_config["bias"][1])
        self.b = resnet_config["b"]
        # if resnet_config["replace_stride_with_dilation"] is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # parameters to set in config
        self.groups = resnet_config["groups"]
        self.short_cat = resnet_config["short_cat"]
        self.base_width = resnet_config["base_width"]

        self.conv1 = BcosConv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, b=self.b)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.fc = BcosConv2d(512 * block.expansion, num_classes)
        self.fc = BcosConv2d(512 * block.expansion, self.num_classes, kernel_size=1, max_out=2, b=self.b)  # Adjusted for classification
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if resnet_config["zero_init_residual"]:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, b=self.b),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.short_cat, b=self.b))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, b=self.b))
        return nn.Sequential(*layers)


    def _resnet_impl(self, x):
        '''Kai: Basically the forward implementation up to the point of the avgpool and fc layer.'''
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        # Freeze all layers except the fc layer
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters

        for param in self.fc.parameters():
            param.requires_grad = True  # Unfreeze the fc layer

    def features(self, inp):
        x = self._resnet_impl(inp)
        return x

    def classifier(self, features):
        '''The prediction head, consisting of avg pool and fc'''
        # old standard RESNET Implementation
        # x = self.avgpool(features)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.fc(features)
        pooling = MyAdaptiveAvgPool2d((1, 1))
        x = pooling.forward(in_tensor = x)
        final = FinalLayer(bias = self.bias, norm = self.log_temperature)
        x = final.forward(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        # x = x.squeeze()
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        if self.num_classes == 1:
            x = x.squeeze()  # Removes dimensions of size 1, resulting in shape [16]
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out
