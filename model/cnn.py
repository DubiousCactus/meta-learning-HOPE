#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Custom ResNet and MobileNet
"""

from HOPE.models.resnet import resnet10, model_urls
from collections import OrderedDict
from typing import Tuple
from torch import Tensor

import torchvision.models as models
import torch.nn.functional as F
import wandb
import torch


class ResNet(torch.nn.Module):
    def __init__(self, model="18", pretrained=True):
        super().__init__()
        if model == "10":
            network = resnet10(num_classes=29*2)
            if pretrained:
                self._load_resnet10_model(network)
        elif model == "18":
            network = models.resnet18(pretrained=pretrained)
        elif model == "34":
            network = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"No models for {model}")
        n_features = network.fc.in_features
        self.resnet = network
        del self.resnet.fc
        wconfig = wandb.config
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, wconfig['experiment.hidden']),
            torch.nn.Dropout(p=wconfig['experiment.dropout'], inplace=True),
            torch.nn.ReLU(),
            torch.nn.Linear(wconfig['experiment.hidden'], 29*2)
        )

    def _load_resnet10_model(self, model: torch.nn.Module):
        res_18_state_dict = torch.hub.load_state_dict_from_url(model_urls["resnet18"])
        # Exclude the fully connected layer
        del res_18_state_dict['fc.weight']
        del res_18_state_dict['fc.bias']
        model.load_state_dict(res_18_state_dict, strict=False)

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Original implementation from PyTorch, modified to return the image features vector.
        '''
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        img_features = x
        x = self.fc(x)

        return x.view(-1, 29, 2), img_features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward_impl(x)


class MobileNet(torch.nn.Module):
    def __init__(self, model="v3-small", pretrained=True):
        super().__init__()
        if model == "v3-small":
            network = models.mobilenet_v3_small(pretrained=pretrained)
        elif model == "v3-large":
            network = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(f"No models for {model}")
        n_features = network.classifier[0].in_features
        self.mobilenet = network
        del self.mobilenet.classifier
        wconfig = wandb.config
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, wconfig['experiment.hidden']),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=wconfig['experiment.dropout'], inplace=True),
            torch.nn.Linear(wconfig['experiment.hidden'], 29*2)
        )

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Original implementation from PyTorch, modified to return the image features vector.
        '''
        x = self.mobilenet.features(x)

        x = self.mobilenet.avgpool(x)
        x = torch.flatten(x, 1)
        img_features = x

        x = self.fc(x)

        return x.view(-1, 29, 2), img_features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward_impl(x)


