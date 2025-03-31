"""
For easier copy-pasting.
No need to extract stuff from a super-coupled codebase :)

Note that if you want to use the pretrained model weights,
then please keep in mind that the outputs might be different from the original implementation
as this uses some slightly different ops than with which the weights were trained.

ResNet-50 implementation modified for B-cos from 
Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb8a2/torchvision/models/resnet.py
"""
import contextlib
import functools
import math
from typing import Any, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import os
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import BACKBONE


# from bcos.bcosconv2d import BcosConv2d
# from bcos.detector_utils import MyAdaptiveAvgPool2d, FinalLayer
import numpy as np
from torch import Tensor

# from data.data_transforms import AddInverse
# from .bcosconv2d import BcosConv2d
from torch.hub import load_state_dict_from_url
# from torch.utils.model_zoo import load_url as load_state_dict_from_url
# from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from bcos import BcosUtilMixin



logger = logging.getLogger(__name__)


class BcosConv2d(nn.Conv2d):
    def __init__(self, *args, b: float = 2.0, **kwargs):
        kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.b = b
        assert self.dilation == (1, 1), "Dilation > 1 is not supported."
        self.detach = False

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        squares = in_tensor**2
        norms = (
            squares.sum(1, keepdim=True)
            if self.groups == 1
            else squares.unflatten(
                1, (self.groups, self.in_channels // self.groups)
            ).sum(2)
        )

        norms = (
            F.avg_pool2d(
                norms,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                divisor_override=1,  # sum, not avg
            )
            + 1e-6  # stabilizing term
        ).sqrt_()

        if self.groups > 1:
            norms = torch.repeat_interleave(
                norms, repeats=self.out_channels // self.groups, dim=1
            )

        return norms

    def set_explanation_mode(self, on: bool):
        self.detach = on

    def forward(self, in_tensor: Tensor) -> Tensor:
        # this is better, as it's from torch + it has a stabilizing term (eps)
        normed_weights = F.normalize(self.weight, dim=(1, 2, 3))  # type: ignore
        out = self._conv_forward(in_tensor, normed_weights, self.bias)

        if self.b == 1:
            return out

        norms = self.calc_patch_norms(in_tensor)

        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norms = norms.detach()

        dynamic_scaling = (maybe_detached_out / norms).abs()
        if self.b != 2:
            dynamic_scaling = (dynamic_scaling + 1e-6).pow_(self.b - 1)

        return dynamic_scaling * out

    # for compatibility with weights
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "linear.weight" in state_dict:
            state_dict[prefix + "weight"] = state_dict.pop(prefix + "linear.weight")
        if prefix + "linear.bias" in state_dict:
            state_dict[prefix + "bias"] = state_dict.pop(prefix + "linear.bias")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class BatchNorm2dUncenteredNoBias(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None
        self.detach = False

    def set_explanation_mode(self, on: bool):
        self.detach = on

    def forward(self, in_tensor: Tensor) -> Tensor:
        if self.training:
            x = in_tensor.detach() if self.detach else in_tensor
            var = x.var(dim=(0, 2, 3), unbiased=False)

            if self.running_var is not None:
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var.detach()
                )
        else:  # evaluation mode
            var = self.running_var

        # might be slightly faster as it avoids division
        rstd = (var + self.eps).rsqrt()[None, ..., None, None]

        result = in_tensor * rstd

        if self.weight is not None:
            result = self.weight[None, ..., None, None] * result
        if self.bias is not None:
            result = result + self.bias[None, ..., None, None]

        return result


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    b: float = 2
):
    """3x3 convolution with padding"""
    return BcosConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        b=b
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    b: float = 2
):
    """1x1 convolution"""
    return BcosConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        b=b
    )


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
        b: float = 2
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3x3(inplanes, planes, stride=stride, b=b)
        self.bn1 = BatchNorm2dUncenteredNoBias(planes)
        self.conv2 = conv3x3(planes, planes, b=b)
        self.bn2 = BatchNorm2dUncenteredNoBias(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        b: float = 2
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, b=b)
        self.bn1 = BatchNorm2dUncenteredNoBias(width)
        self.conv2 = conv3x3(width, width, stride, groups, b=b)
        self.bn2 = BatchNorm2dUncenteredNoBias(width)
        self.conv3 = conv1x1(width, planes * self.expansion, b=b)
        self.bn3 = BatchNorm2dUncenteredNoBias(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


@BACKBONE.register_module(module_name="resnet34_bcos_v2_minimal")
class ResNet34_bcos_v2_minimal(BcosUtilMixin, nn.Module):
    def __init__(
        self, resnet_config
    ):
        super(ResNet34_bcos_v2_minimal, self).__init__()
        self.inplanes = 64
        inplanes = 64
        self.b = resnet_config["b"]
        self.groups = resnet_config["groups"]
        self.base_width = resnet_config["base_width"]


        block= BasicBlock #Type[Union[BasicBlock, Bottleneck]],
        layers=[3, 4, 6, 3] #: List[int],
        self.num_classes = resnet_config["num_classes"]

        small_inputs = False
        if small_inputs:
            self.conv1 = conv3x3(in_chans, self.inplanes, b=self.b)
            self.pool = None
        else:
            in_chans = 6
            self.conv1 = BcosConv2d(
                in_chans, self.inplanes, kernel_size=7, stride=2, padding=3, b=self.b
            )
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = BatchNorm2dUncenteredNoBias(self.inplanes)

        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        try:
            self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
            last_ch = inplanes * 8
        except IndexError:
            self.layer4 = None
            last_ch = inplanes * 4

        self.num_features = last_ch * block.expansion
        # self.num_features = 4096

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = BcosConv2d(
            self.num_features,
            self.num_classes,
            kernel_size=1,
            b=self.b
        )
        self.logit_bias = (
            resnet_config["logit_bias"]
            if resnet_config["logit_bias"] is not None
            else math.log(1 / (self.num_classes - 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, b=self.b),
                BatchNorm2dUncenteredNoBias(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, b=self.b
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    b=self.b
                )
            )
        return nn.Sequential(*layers)

    

    def features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        if self.pool is not None:
            x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x
    

    def classifier(self, features) -> Tensor:
        # x = self.forward_features(x)
        x = self.fc(features)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = x + self.logit_bias
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
        elif isinstance(module, (Bottleneck, BasicBlock, BcosConv2d)):
            for submodule in module.children():
                self.initialize_weights(submodule)
        # Ignore activation, pooling, and sequential layers
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            pass  # Do nothing
        else:
            print(f'unknown module type {type(module)}')

    def freeze(self):
        # Freeze all layers except the fc layer
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters

        for param in self.fc.parameters():
            param.requires_grad = True  # Unfreeze the fc layer


    @contextlib.contextmanager
    def explanation_mode(self):
        for m in self.modules():
            if hasattr(m, "set_explanation_mode"):
                m.set_explanation_mode(True)
        yield
        for m in self.modules():
            if hasattr(m, "set_explanation_mode"):
                m.set_explanation_mode(False)


# def _resnet(
#     arch: str,
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     pretrained: bool = False,
#     progress: bool = True,
#     inplanes: int = 64,
#     **kwargs: Any,
# ) -> BcosResNet:
#     model = BcosResNet(block, layers, inplanes=inplanes, **kwargs)
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             URLS[arch],
#             progress=progress,
#         )
#         # load with changed keys
#         model.load_state_dict(state_dict)
#     return model


# # ---------------------
# # ResNets for ImageNet
# # ---------------------
# def resnet18(
#     pretrained: bool = False, progress: bool = True, **kwargs: Any
# ) -> BcosResNet:
#     return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


# def resnet34(
#     pretrained: bool = False, progress: bool = True, **kwargs: Any
# ) -> BcosResNet:
#     return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


# def resnet50(
#     pretrained: bool = False,
#     progress: bool = True,
#     long_version: bool = True,  # whether to use the long trained weights
#     **kwargs: Any,
# ) -> BcosResNet:
#     name = "resnet50_long" if long_version else "resnet50"
#     return _resnet(name, Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


# def resnet101(
#     pretrained: bool = False, progress: bool = True, **kwargs: Any
# ) -> BcosResNet:
#     return _resnet(
#         "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
#     )


# def resnet152(
#     pretrained: bool = False,
#     progress: bool = True,
#     long_version: bool = True,  # whether to use the long trained weights
#     **kwargs: Any,
# ) -> BcosResNet:
#     name = "resnet152_long" if long_version else "resnet152"
#     return _resnet(name, Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


# def resnext50_32x4d(
#     pretrained: bool = False, progress: bool = True, **kwargs: Any
# ) -> BcosResNet:
#     kwargs.setdefault("groups", 32)
#     kwargs.setdefault("width_per_group", 4)
#     return _resnet(
#         "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
#     )


# # ---------------------
# # Model URLs
# # ---------------------
# BASE = "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights"
# URLS = {
#     "resnet18": f"{BASE}/resnet_18-68b4160fff.pth",
#     "resnet34": f"{BASE}/resnet_34-a63425a03e.pth",
#     "resnet50": f"{BASE}/resnet_50-ead259efe4.pth",
#     "resnet101": f"{BASE}/resnet_101-84c3658278.pth",
#     "resnet152": f"{BASE}/resnet_152-42051a77c1.pth",
#     "resnext50_32x4d": f"{BASE}/resnext_50_32x4d-57af241ab9.pth",
#     "resnet50_long": f"{BASE}/resnet_50_long-ef38a88533.pth",
#     "resnet152_long": f"{BASE}/resnet_152_long-0b4b434939.pth",
# }