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
from collections import OrderedDict


class HOPENet(torch.nn.Module):
    def __init__(self, resnet_path: str, graphnet_path: str, graphunet_path: str):
        super().__init__()
        self.resnet = resnet10(pretrained=False, num_classes=29 * 2)
        self.graphnet = GraphNet(in_features=514, out_features=2)
        self.graphunet = GraphUNet(in_features=2, out_features=3)
        if resnet_path:
            print(f"[*] Loading ResNet state dict form {resnet_path}")
            self._load_state_dict(self.resnet, resnet_path)
        else:
            print("[!] ResNet is randomly initialized!")
        if graphnet_path:
            print(f"[*] Loading GraphNet state dict form {graphnet_path}")
            self._load_state_dict(self.graphnet, graphnet_path)
        else:
            print("[!] GraphNet is randomly initialized!")
        if graphunet_path:
            print(f"[*] Loading GraphUNet state dict form {graphunet_path}")
            self._load_state_dict(self.graphunet, graphunet_path)
        else:
            print("[!] GraphUNet is randomly initialized!")

    def _load_state_dict(self, module, resnet_path):
        """
        Load a state_dict in a module agnostic way (when DataParallel is used, the module is
        wrapped and the saved state dict is not applicable to non-wrapped modules).
        """
        ckpt = torch.load(resnet_path)
        new_state_dict = OrderedDict()
        for k, v in ckpt["model_state_dict"].items():
            if "module" in k:
                k = k.replace("module.", "")
            new_state_dict[k] = v
        module.load_state_dict(new_state_dict)

    def forward(self, x):
        points2D_init, features = self.resnet(x)
        features = features.unsqueeze(1).repeat(1, 29, 1)
        # batch = points2D.shape[0]
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)
        points3D = self.graphunet(points2D)
        return points2D_init, points2D, points3D
