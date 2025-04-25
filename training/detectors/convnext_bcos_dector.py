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
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks.base import BACKBONE
from loss import LOSSFUNC

from convnext import convnext_tiny

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='convnext_bcos')
class Convnext_Bcos_Detector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        if config['pretrained'] == None:
            backbone.apply(backbone.initialize_weights)
            logger.info("Initialized backbone weights from scratch!")
            return backbone
        
        # only use to print the model to compare
        """ print("-------- Summary backbone")
        summary(backbone, depth=5, input_size=(1, 6, 224, 224))

        model = convnext_tiny(pretrained=False, num_classes=2)
        print("-------- Summary original implementation")
        summary(model, depth=5, input_size=(1, 6, 224, 224)) """
        # if donot load the pretrained weights, fail to get good results
        # self.block_setting = config['CNBlockConfig'],
        #self.stochastic_depth_prob = config['',
        # layer_scale: float = 1e-6,
        # num_classes: int = 1000,
        # in_chans: int = 6,
        # block: Optional[Callable[..., nn.Module]] = None,
        # conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
        # logit_bias: Optional[float] = None,
        # logit_temperature: Optional[float] = None,
        # **kwargs: Any,
        ## CHANGE THIS HERE FOR THE CLUSTER TO NOT MAP LOCATION CPU
        """ if config['pretrained'] != 'None':
            state_dict = torch.load(config['pretrained'], map_location=torch.device('cpu'))
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!') """
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier_impl(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap, rc, f1 = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'rc':rc, 'f1':f1}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
