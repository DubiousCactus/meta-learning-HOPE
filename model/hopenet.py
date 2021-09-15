#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Custom HOPE-Net
"""

import torch

from HOPE.models.graphunet import GraphUNet, GraphNet
from model.graphnet import GraphUNetBatchNorm
from model.cnn import ResNet, MobileNet
from model.wrapper import InitWrapper
from util.utils import load_state_dict


class HOPENet(InitWrapper, torch.nn.Module):
    def __init__(
        self, cnn_def: str, resnet_path: str, graphnet_path: str, graphunet_path: str
    ):
        super().__init__()
        cnn_def = cnn_def.lower()
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        self.resnet = cnn
        self.graphnet = GraphNet(in_features=514, out_features=2)
        self.graphunet = GraphUNetBatchNorm(in_features=2, out_features=3)
        if resnet_path:
            print(f"[*] Loading ResNet state dict from {resnet_path}")
            self.randomly_initialize_weights = False
            load_state_dict(self.resnet, resnet_path)
            print(f"[*] Freezing ResNet parameters!")
            for p in self.resnet.parameters():
                p.requires_grad = False
        else:
            print("[!] ResNet is randomly initialized!")
        if graphnet_path:
            print(f"[*] Loading GraphNet state dict from {graphnet_path}")
            self.randomly_initialize_weights = False
            load_state_dict(self.graphnet, graphnet_path)
        else:
            print("[!] GraphNet is randomly initialized!")
        if graphunet_path:
            print(f"[*] Loading GraphUNet state dict from {graphunet_path}")
            self.randomly_initialize_weights = False
            load_state_dict(self.graphunet, graphunet_path)
        else:
            print("[!] GraphUNet is randomly initialized!")

    def forward(self, x, gt_2d=None):
        if gt_2d is None:
            points2D_init, features = self.resnet(x)
            features = features.unsqueeze(1).repeat(1, 29, 1)
            in_features = torch.cat([points2D_init, features], dim=2)
            points2D = self.graphnet(in_features)
        else:
            points2D_init = gt_2d
            points2D = gt_2d
        points3D = self.graphunet(points2D)
        return points2D_init, points2D, points3D


class GraphNetwResNet(InitWrapper, torch.nn.Module):
    def __init__(self, cnn_def: str, resnet_path: str):
        super().__init__()
        cnn_def = cnn_def.lower()
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        self.resnet = cnn
        self.graphnet = GraphNet(in_features=514, out_features=2)
        if resnet_path:
            print(f"[*] Loading ResNet state dict form {resnet_path}")
            self.randomly_initialize_weights = False
            load_state_dict(self.resnet, resnet_path)
        else:
            print("[!] ResNet is randomly initialized!")

    def forward(self, x):
        points2D_init, features = self.resnet(x)
        features = features.unsqueeze(1).repeat(1, 29, 1)
        # batch = points2D.shape[0]
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)
        return points2D_init, points2D
