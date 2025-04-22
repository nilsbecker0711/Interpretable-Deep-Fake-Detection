'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py

tilo: the architecture has been modified to make it a b-cos model
'''

import os
import argparse
import logging

import math
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

from bcos.modules import BcosConv2d, LogitLayer
from bcos.common import BcosUtilMixin
from bcos.modules import norms


import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

DEFAULT_NORM_LAYER = norms.NoBias(norms.DetachablePositionNorm2d)


class SeparableConv2d(nn.Module, BcosUtilMixin):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # depthwise as B‑cos
        self.conv1 = BcosConv2d(in_channels, in_channels, kernel_size, 
                                stride, padding, dilation,
                                groups=in_channels, bias=False)

        # pointwise as B‑cos
        self.pointwise = BcosConv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module, BcosUtilMixin):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = BcosConv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = DEFAULT_NORM_LAYER(out_filters)
        else:
            self.skip = None

        #self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            #rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(DEFAULT_NORM_LAYER(out_filters))
            filters = out_filters

        for i in range(reps-1):
            #rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(DEFAULT_NORM_LAYER(filters))

        if not grow_first:
            #rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(DEFAULT_NORM_LAYER(out_filters))

        """ if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False) """

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


@BACKBONE.register_module(module_name="xception_bcos")
class XceptionBcos(BcosUtilMixin, nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(XceptionBcos, self).__init__()
        # Initialize components from the mixin if necessary
        BcosUtilMixin.__init__(self,)# **kwargs)

        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        self.logit_bias = xception_config["logit_bias"]
        self.logit_temperature = xception_config["logit_temperature"]
        inc = xception_config["in_chans"]
        #dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = BcosConv2d(inc, 32, 3, 2, 0, bias=False)
        
        self.bn1 = DEFAULT_NORM_LAYER(32)
        #self.relu = nn.ReLU(inplace=True)

        self.conv2 = BcosConv2d(32, 64, 3, bias=False)
        self.bn2 = DEFAULT_NORM_LAYER(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = DEFAULT_NORM_LAYER(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = DEFAULT_NORM_LAYER(2048)
        # used for iid
        final_channel = 2048
        if self.mode == 'adjust_channel_iid':
            final_channel = 512
            self.mode = 'adjust_channel'
        """ self.last_linear = nn.Linear(final_channel, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(final_channel, self.num_classes)
            ) """
        
        self.classifier_head = nn.Sequential(
            DEFAULT_NORM_LAYER(final_channel), # B‑cos normalize
            BcosConv2d(final_channel, self.num_classes, 1, bias=False),  # 1×1 B‑cos conv
        )
        self.logit_layer = LogitLayer(
            logit_temperature=self.logit_temperature,
            logit_bias=self.logit_bias or -math.log(self.num_classes - 1),
        )


        self.adjust_channel = nn.Sequential(
            BcosConv2d(2048, 512, 1, 1),
            DEFAULT_NORM_LAYER(512)
            #nn.ReLU(inplace=False),
        )
        """ for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


           
    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)  

        return x

    def fea_part1_1(self, x):  
        
        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.relu(x) 

        return x
    
    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)     
        
        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.relu(x) 

        return x
    
    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        #x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x
     
    def features(self, input):
        x = self.fea_part1(input)    

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        
        return x

    """ def classifier(self, features,id_feat=None):
        # for iid
        if self.mode == 'adjust_channel':
            x = features
        else:
            #x = self.relu(features)

        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        self.last_emb = x
        # for iid
        if id_feat!=None:
            out = self.last_linear(x-id_feat)
        else:
            out = self.last_linear(x)
        return out """
    
    def classifier(self, features, id_feat=None):
        # features: [B, C, H, W], already B‑cos transformed
        x = features

        # run through our B‑cos classifier conv
        x = self.classifier_head(x) # [B, num_classes, H, W]

        # global average pool to [B, num_classes]
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # optionally subtract an identity feature (for iid mode)
        if id_feat is not None:
            x = x - id_feat

        # finally apply the learned logit bias & temperature
        logits = self.logit_layer(x) # [B, num_classes]
        return logits


    def forward(self, input, id_feat=None):
        x = self.features(input)
        out = self.classifier(x, id_feat)
        return out, x

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Recursively apply to custom modules
        elif isinstance(module, SeparableConv2d) or isinstance(module, Block) or isinstance(module, BcosConv2d):
            for submodule in module.children():
                self.initialize_weights(submodule)
        # Ignore activation, pooling, and sequential layers
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            pass  # Do nothing
        else:
            print(f'unknown module type {type(module)}')
    
    @contextlib.contextmanager
    def explanation_mode(self):
        for m in self.modules():
            if hasattr(m, "set_explanation_mode"):
                m.set_explanation_mode(True)
        yield
        for m in self.modules():
            if hasattr(m, "set_explanation_mode"):
                m.set_explanation_mode(False)
