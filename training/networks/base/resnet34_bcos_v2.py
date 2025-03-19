'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for ResNet34 backbone.
'''
import math
import os
import logging
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torchvision
from torchvision.ops import StochasticDepth
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import BACKBONE
import contextlib

from bcos.modules import BcosConv2d, LogitLayer, norms
from bcos.common import BcosUtilMixin
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


DEFAULT_NORM_LAYER = norms.NoBias(norms.DetachablePositionNorm2d)
#DEFAULT_NORM_LAYER = norms.BatchNorm2dUncenteredNoBias
DEFAULT_CONV_LAYER = BcosConv2d

logger = logging.getLogger(__name__)

def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    b: float = 2
):
    """3x3 convolution with padding"""
    return conv_layer(
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
    conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    b: float = 2
):
    """1x1 convolution"""
    return conv_layer(
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
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
        b: float = 2
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=stride, conv_layer=conv_layer, b=b)
        self.bn1 = norm_layer(planes)
        #self.bn1 = BatchNorm2dUncenteredNoBias(planes)
        self.conv2 = conv3x3(planes, planes, conv_layer=conv_layer, b=b)
        #self.bn2 = BatchNorm2dUncenteredNoBias(planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)
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
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
        b: float = 2
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_layer=conv_layer, b=b)
        # self.bn1 = BatchNorm2dUncenteredNoBias(width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, conv_layer=conv_layer, b=b)
        # self.bn2 = BatchNorm2dUncenteredNoBias(width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_layer=conv_layer, b=b)
        # self.bn3 = BatchNorm2dUncenteredNoBias(planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.act(out)

        return out

@BACKBONE.register_module(module_name="resnet34_bcos_v2")
class ResNet34_bcos_v2(BcosUtilMixin, nn.Module):

    def __init__(self, resnet_config):
        super(ResNet34_bcos_v2, self).__init__()

        # Initialize components from the mixin if necessary
        BcosUtilMixin.__init__(self,)# **kwargs)

        """ Constructor
        Args:
            resnet_config: configuration file with the dict format
        """
        self.inplanes = 64 # KAI: can be set to a different dynamical value
        # KAI: need to define this as well for the layers, since it is otherwise updated 
        # in each call of self._make_layer...
        inplanes = 64 
        self.b = resnet_config['b']
        #self.mode = resnet_config["mode"]
        self.groups = resnet_config["groups"]
        self.base_width = resnet_config["base_width"]

        self.num_classes = resnet_config["num_classes"]
        block = BasicBlock
        layers = [3, 4, 6, 3]
        n = len(layers)  # number of stages
        #self.short_cat = resnet_config["short_cat"]
        self.dilation = 1

        self._norm_layer = DEFAULT_NORM_LAYER
        norm_layer = DEFAULT_NORM_LAYER
        self._conv_layer = DEFAULT_CONV_LAYER
        conv_layer = DEFAULT_CONV_LAYER
        # ------------- NEW ------------------

        # if kwargs:
        #     print("The following args passed to model will be ignored", kwargs)

        # if "norm_layer" in resnet_config.keys():
        #     norm_layer = resnet_config["norm_layer"]
        # else:
        #     norm_layer = nn.Identity
        #     norm_layer = DEFAULT_NORM_LAYER
        # if "conv_layer" in resnet_config.keys():
        #     self._conv_layer = resnet_config["conv_layer"]
        # else:
        # self._act_layer = act_layer

        # if resnet_config["replace_stride_with_dilation"] is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead

        replace_stride_with_dilation = [False] * (n - 1)
        if len(replace_stride_with_dilation) != n - 1:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a {n - 1}-element tuple, got {replace_stride_with_dilation}"
            )
        
        in_chans = resnet_config["in_chans"]
        stochastic_depth_prob = resnet_config["stochastic_depth_prob"]

        if resnet_config["small_inputs"]:
            self.conv1 = conv3x3(
                in_chans,
                self.inplanes,
                conv_layer=conv_layer,
                b = self.b
            )
            self.pool = None
        else:
            self.conv1 = conv_layer(
                in_chans,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                b = self.b
            )
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = norm_layer(self.inplanes)
        # self.bn1 = BatchNorm2dUncenteredNoBias(self.inplanes)
        # self.act = act_layer(inplace=True)

        self.__total_num_blocks = sum(layers)
        self.__num_blocks = 0
        self.layer1 = self._make_layer(
            block,
            inplanes,
            layers[0],
            stochastic_depth_prob=stochastic_depth_prob,
            b = self.b
        )
        self.layer2 = self._make_layer(
            block,
            inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            stochastic_depth_prob=stochastic_depth_prob,
            b = self.b
        )
        print(self.inplanes)
        self.layer3 = self._make_layer(
            block,
            inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            stochastic_depth_prob=stochastic_depth_prob,
            b = self.b
        )
        try:
            self.layer4 = self._make_layer(
                block,
                inplanes * 8,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
                stochastic_depth_prob=stochastic_depth_prob,
                b = self.b
            )
            last_ch = inplanes * 8
        except IndexError:
            self.layer4 = None
            last_ch = inplanes * 4

        self.num_features = last_ch * block.expansion #(4096 * 4) * 1
        #self.num_features = self.inplanes  # 4096
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = conv_layer(
            self.num_features,
            self.num_classes,
            kernel_size=1,
            b=self.b
        )
        # logit_bias = None
        self.logit_bias = (
            resnet_config["logit_bias"]
            if resnet_config["logit_bias"] is not None
            else math.log(1 / (self.num_classes - 1))
        )
        logit_temperature = None
        # self.fc = BcosConv2d(512 * block.expansion, self.num_classes, kernel_size=1, max_out=2)  # Adjusted for classification
        self.logit_layer = LogitLayer(
            logit_temperature=resnet_config["logit_temperature"],
            logit_bias=self.logit_bias or -math.log(self.num_classes - 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if resnet_config["zero_init_residual"]:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        stochastic_depth_prob: float = 0.0,
        b: float= 1.25,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        conv_layer = self._conv_layer
        # act_layer = self._act_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    conv_layer=conv_layer,
                    b=b 
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
                # act_layer=act_layer,
                stochastic_depth_prob=stochastic_depth_prob
                * self.__num_blocks
                / (self.__total_num_blocks - 1),
                b=b,
            )
        )
        self.__num_blocks += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer,
                    # act_layer=act_layer,
                    stochastic_depth_prob=stochastic_depth_prob
                    * self.__num_blocks
                    / (self.__total_num_blocks - 1),
                    b=b,
                )
            )
            self.__num_blocks += 1

        return nn.Sequential(*layers)


    def features(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        # x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x

    def classifier(self, features):
        '''The prediction head, consisting of avg pool and fc'''
        # old standard RESNET Implementation
        # x = self.avgpool(features)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # x = F.adaptive_avg_pool2d(features, (1, 1))  # Global average pooling
        # x = self.fc(features)
        # # x = x.squeeze()
        # # x = x.view(x.size(0), -1)  # Flatten the tensor
        # if self.num_classes == 1:
        #     x = x.squeeze()  # Removes dimensions of size 1, resulting in shape [16]

        x = self.fc(features)
        x = self.avgpool(x)
        x = x.flatten(1)
        # x = x + self.logit_bias
        x = self.logit_layer(x)
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
