'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the ResnetDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{wang2020cnn,
  title={CNN-generated images are surprisingly easy to spot... for now},
  author={Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={8695--8704},
  year={2020}
}

Notes:
We chose to use ResNet-34 as the backbone instead of ResNet-50 because the number of parameters in ResNet-34 is relatively similar to that of Xception. This similarity allows us to make a more meaningful and fair comparison between different architectures.
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks.base import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='resnet34_bcos')
class ResnetBcosDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        if config['pretrained'] == None:
            for m in backbone.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            logger.info("Initialized backbone weights from scratch!")
            return backbone
        else:
            #FIXME: current load pretrained weights only from the backbone, not here
            # # if donot load the pretrained weights, fail to get good results
            state_dict = torch.load(config['pretrained'], weights_only=False)
            # state_dict = {'resnet.'+k:v for k, v in state_dict.items() if 'fc' not in k}
            # backbone.load_state_dict(state_dict, False)
            if 'resnet34-333f7ec4.pth' in str(config['pretrained']):# kai: handle the ImageNet weights differently, 
                adapted_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace("conv", "conv.linear").replace("fc", "fc.linear")
                    if new_key in backbone.state_dict() and backbone.state_dict()[new_key].shape == value.shape:
                        adapted_state_dict[new_key] = value
                backbone.load_state_dict(adapted_state_dict, strict=False)
                # handle the prediction head, which is not inititalized otherwise
                # nn.init.kaiming_normal_(backbone.fc.linear.weight)
                # if backbone.fc.linear.bias is not None:
                #     backbone.fc.linear.bias.data.zero_()
                nn.init.xavier_uniform_(backbone.fc.linear.weight)
                if backbone.fc.linear.bias is not None:
                    nn.init.constant_(backbone.fc.linear.bias, 0) 
            else:
                backbone.load_state_dict(state_dict, strict=False)
            logger.info('Load pretrained model successfully!')
            return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap, rc, f1 = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'rc': rc, 'f1': f1}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)

        # features = F.adaptive_avg_pool2d(features, (1, 1))
        # features = torch.flatten(features, 1)
        # features = features[..., None, None]
        
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        # pred = torch.clamp(pred, min=-100, max=100)
        # print(pred)
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

    def debug_weights_and_features(self, data_dict: dict):
        logger.info("=== Debugging Weights and Features ===")
        
        # logger.info(all weight values
        logger.info("\n--- Model Weights ---")
        # for name, param in self.backbone.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"{name}:\n{param.data}")
        
        # Get features
        features = self.features(data_dict)
        
        # Get classifier output
        pred = self.classifier(features)
        
        # Get probability values
        pred = torch.clamp(pred, min=-100, max=100)
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        for name, param in self.model.backbone.named_parameters():
            # self.logger.info(f"Updated Weights - {name}: {param.data}")
            if param.requires_grad:
                self.logger.info(f"{name} - mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}, min: {param.data.min().item():.6f}, max: {param.data.max().item():.6f}, lowest_abs: {param.data.abs().min().item():.6e}, contains_nan: {not torch.isfinite(param.data).all()}")
        
        
        self.logger.info('-------------------- INPUT Values ----------------')
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                min_val = value.min().item()
                max_val = value.max().item()
                mean_val = value.mean().item()
                std_val = value.std().item()
                lowest_abs_val = value.abs().min().item()
                contains_nan = not torch.isfinite(value).all()

                self.logger.info(f"Input {key} - mean: {mean_val:.6f}, std: {std_val:.6f}, min: {min_val:.6f}, max: {max_val:.6f}, lowest_abs: {lowest_abs_val:.6e}, contains_nan: {contains_nan}")
        
        logger.info("\n--- Extracted Features ---")
        logger.info(features)
        logger.info("\n--- Classifier Output ---")
        logger.info(pred)
        logger.info("\n--- Prediction Probabilities ---")
        logger.info(prob)
        logger.info("=== Debugging Complete ===")
        return {"cls": pred, "prob": prob, "feat": features}

