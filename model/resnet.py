#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Custom ResNet
"""

from collections import OrderedDict
from typing import Tuple
from torch import Tensor

import torchvision.models as models
import torch


class ResNet(torch.nn.Module):
    def __init__(self, model="resnet18", pretrained=True):
        super().__init__()
        if model == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif model == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"No models for {model}")
        n_features = resnet.fc.in_features
        self.resnet = resnet
        del self.resnet.fc
        self.fcl = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_features//2),
            torch.nn.Dropout(p=0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(n_features//2, 29*2)
        )

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Original implementation from PyTorch
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
        x = self.fcl(x)

        return x.view(-1, 29, 2), img_features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward_impl(x)
