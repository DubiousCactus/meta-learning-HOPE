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
from HOPE.models.resnet import resnet10


class HOPENet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet10(pretrained=False, num_classes=29*2)
        self.graphnet = GraphNet(in_features=514, out_features=2)
        self.graphunet = GraphUNet(in_features=2, out_features=3)

    def forward(self, x):
        points2D_init, features = self.resnet(x)
        features = features.unsqueeze(1).repeat(1, 29, 1)
        # batch = points2D.shape[0]
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)
        points3D = self.graphunet(points2D)
        return points2D_init, points2D, points3D

