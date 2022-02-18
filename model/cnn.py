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
from typing import Tuple, Union
from torch import Tensor

import torchvision.models as models
import learn2learn as l2l
import torch

from model.wrapper import InitWrapper


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        torch.nn.init.constant_(m.bias.data, 0)


class ResNet12(InitWrapper, torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.randomly_initialize_weights = False
        hidden1, hidden2 = 256, 128
        n_features = 125440
        self._n_features = n_features
        self.resnet = l2l.vision.models.ResNet12Backbone(avg_pool=False, wider=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden1),
            # torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1, 29 * 3),
        )
        self.fc.apply(initialize_weights)
        import torchsummary
        torchsummary.summary(self.resnet.to('cuda'), input_size=(3, 224, 224))

    def _forward_impl(self, x: Tensor, features_only=False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        f = self.resnet(x)
        f = torch.flatten(f, 1)

        if features_only:
            return f
        else:
            x = self.fc(f)
            return (x.view(-1, 29, 3), f)

    def forward(self, x: Tensor, features_only=False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        return self._forward_impl(x, features_only=features_only)

    @property
    def out_features(self) -> int:
        return self._n_features

    @property
    def features(self) -> Lambda:
        return Lambda(lambda x: self(x, features_only=True))

    @property
    def head(self):
        return self.fc



class ResNet(InitWrapper, torch.nn.Module):
    def __init__(self, model="18", pretrained=True, hand_only=True):
        super().__init__()
        self.randomly_initialize_weights = False
        self._dim = 21 if hand_only else 29
        hidden = 128
        if model == "10":
            network = resnet10(num_classes=21 * 3)
            if pretrained:
                self._load_resnet10_model(network)
        elif model == "18":
            network = models.resnet18(pretrained=pretrained)
            hidden = 256
        elif model == "34":
            network = models.resnet34(pretrained=pretrained)
            hidden = 256
        elif model == "50":
            network = models.resnet50(pretrained=pretrained)
            hidden = 512
        else:
            raise ValueError(f"No models for {model}")
        n_features = network.fc.in_features
        self._n_features = n_features
        self.resnet = network
        del self.resnet.fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, self._dim * 3),
        )
        self.fc.apply(initialize_weights)

    def _load_resnet10_model(self, model: torch.nn.Module):
        res_18_state_dict = torch.hub.load_state_dict_from_url(model_urls["resnet18"])
        # Exclude the fully connected layer
        del res_18_state_dict["fc.weight"]
        del res_18_state_dict["fc.bias"]
        model.load_state_dict(res_18_state_dict, strict=False)

    def _forward_impl(self, x: Tensor, features_only=False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Original implementation from PyTorch, modified to return the image features vector.
        """
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
        features = torch.flatten(x, 1)

        if features_only:
            return features
        else:
            x = self.fc(features)
            return (x.view(-1, self._dim, 3), features)

    def forward(self, x: Tensor, features_only=False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        return self._forward_impl(x, features_only=features_only)

    @property
    def out_features(self) -> int:
        return self._n_features

    @property
    def features(self) -> Lambda:
        return Lambda(lambda x: self(x, features_only=True))

    @property
    def head(self):
        return self.fc


class MobileNet(InitWrapper, torch.nn.Module):
    def __init__(self, model="v3-small", pretrained=True):
        super().__init__()
        self.randomly_initialize_weights = False
        if model == "v3-small":
            network = models.mobilenet_v3_small(pretrained=pretrained)
        elif model == "v3-large":
            network = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(f"No models for {model}")
        n_features = network.classifier[0].in_features
        self.mobilenet = network
        del self.mobilenet.classifier
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Hardswish(),
            torch.nn.Linear(n_features, 29 * 2),
        )

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Original implementation from PyTorch, modified to return the image features vector.
        """
        x = self.mobilenet.features(x)

        x = self.mobilenet.avgpool(x)
        x = torch.flatten(x, 1)
        img_features = x

        x = self.fc(x)

        return x.view(-1, 29, 2), img_features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward_impl(x)

    @property
    def features(self):
        return torch.nn.Sequential(
            self.mobilenet, Lambda(lambda x: x.view(-1, self._n_features))
        )

    @property
    def head(self):
        return self.fc
