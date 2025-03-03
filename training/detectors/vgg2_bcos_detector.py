import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
from networks.base.vgg2_bcos import vgg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.hub import load_state_dict_from_url

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks.base import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='vgg19_v2_bcos')
class VGGBcosDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        print("Registrierte Backbones:", BACKBONE.data.keys())
        torch.cuda.empty_cache()
        # prepare the backbone
        try:
            backbone_class = BACKBONE[config['backbone_name']]
            model_config = config['backbone_config']
            backbone = backbone_class(model_config)
        except KeyError:
            vgg(config)
            backbone_class = BACKBONE[config['backbone_name']]
            model_config = config['backbone_config']
            backbone = backbone_class(model_config)
        pretrained = config["pretrained"]
        
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
            del state_dict["classifier.6.weight"]
            del state_dict["classifier.6.bias"]
        
            backbone.load_state_dict(state_dict, strict=False)
            logger.info('Load pretrained model successfully!')
            return backbone
        else:
            return backbone
        
    
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier2(self, features: torch.tensor) -> torch.tensor:
        #print(features.shape)
        print(self.backbone.classifier(features))
        return self.backbone.classifier(features)
    
    def classifier(self, features: torch.tensor) -> torch.tensor: #UNUSED!!!
        print(features.shape)
        pred = self.backbone.classifier(features)  # [32, 2, 7, 7]
        #pred = F.adaptive_avg_pool2d(pred, 1)  # Pool to [32, 2, 1, 1]
        #pred = pred.view(pred.shape[0], -1)  # Flatten to [32, 2]
        return pred
        
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls'] 
        #pred = F.adaptive_avg_pool2d(pred, 1)
        #pred = pred.view(pred.shape[0], -1)  # Flatten to [32, 2]
        #print(pred.shape)
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred = F.adaptive_avg_pool2d(pred, 1)  # Pool to [32, 2, 1, 1]
        pred = pred.view(pred.shape[0], -1)  # Flatten to [32, 2]
        print(pred.shape)
        # compute metrics for batch data
        auc, eer, acc, ap, rc, f1 = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'rc': rc, 'f1': f1}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        #print(features.shape)
        pred = self.classifier2(features)
        pred = torch.clamp(pred, min=-100, max=100)
        prob = torch.softmax(pred, dim = 1)[:, 1]
        # prob = torch.sigmoid(pred)
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

