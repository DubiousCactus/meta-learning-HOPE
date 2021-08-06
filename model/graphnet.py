#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Modified GraphUNet that uses Batch Norm layers.
"""

from HOPE.models.graphunet import GraphPool, GraphUnpool, GraphConv
from torch.nn.parameter import Parameter

import torch


class GraphUNetBatchNorm(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3):
        super(GraphUNetBatchNorm, self).__init__()

        self.A_0 = Parameter(torch.eye(29).float().cuda(), requires_grad=True)
        self.A_1 = Parameter(torch.eye(15).float().cuda(), requires_grad=True)
        self.A_2 = Parameter(torch.eye(7).float().cuda(), requires_grad=True)
        self.A_3 = Parameter(torch.eye(4).float().cuda(), requires_grad=True)
        self.A_4 = Parameter(torch.eye(2).float().cuda(), requires_grad=True)
        self.A_5 = Parameter(torch.eye(1).float().cuda(), requires_grad=True)

        self.gconv1 = GraphConv(in_features, 4)  # 29 = 21 H + 8 O
        self.bn1 = torch.nn.BatchNorm1d(29)
        self.pool1 = GraphPool(29, 15)

        self.gconv2 = GraphConv(4, 8)  # 15 = 11 H + 4 O
        self.bn2 = torch.nn.BatchNorm1d(15)
        self.pool2 = GraphPool(15, 7)

        self.gconv3 = GraphConv(8, 16)  # 7 = 5 H + 2 O
        self.bn3 = torch.nn.BatchNorm1d(7)
        self.pool3 = GraphPool(7, 4)

        self.gconv4 = GraphConv(16, 32)  # 4 = 3 H + 1 O
        self.bn4 = torch.nn.BatchNorm1d(4)
        self.pool4 = GraphPool(4, 2)

        self.gconv5 = GraphConv(32, 64)  # 2 = 1 H + 1 O
        self.bn5 = torch.nn.BatchNorm1d(2)
        self.pool5 = GraphPool(2, 1)

        self.fc1 = torch.nn.Linear(64, 20)
        self.bn6 = torch.nn.BatchNorm1d(1)
        self.bn6p = torch.nn.BatchNorm1d(1)
        self.fc2 = torch.nn.Linear(20, 64)

        self.unpool6 = GraphUnpool(1, 2)
        self.gconv6 = GraphConv(128, 32)

        self.unpool7 = GraphUnpool(2, 4)
        self.gconv7 = GraphConv(64, 16)

        self.unpool8 = GraphUnpool(4, 7)
        self.gconv8 = GraphConv(32, 8)

        self.unpool9 = GraphUnpool(7, 15)
        self.gconv9 = GraphConv(16, 4)

        self.unpool10 = GraphUnpool(15, 29)
        self.gconv10 = GraphConv(8, out_features, activation=None)

        self.ReLU = torch.nn.ReLU()

    def _get_decoder_input(self, X_e, X_d):
        return torch.cat((X_e, X_d), 2)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_0)
        X_0 = self.bn1(X_0)
        X_1 = self.pool1(X_0)

        X_1 = self.gconv2(X_1, self.A_1)
        X_1 = self.bn2(X_1)
        X_2 = self.pool2(X_1)

        X_2 = self.gconv3(X_2, self.A_2)
        X_2 = self.bn3(X_2)
        X_3 = self.pool3(X_2)

        X_3 = self.gconv4(X_3, self.A_3)
        X_3 = self.bn4(X_3)
        X_4 = self.pool4(X_3)

        X_4 = self.gconv5(X_4, self.A_4)
        X_4 = self.bn5(X_4)
        X_5 = self.pool5(X_4)

        global_features = self.ReLU(self.fc1(X_5))
        global_features = self.bn6(global_features)
        global_features = self.ReLU(self.fc2(global_features))
        global_features = self.bn6p(global_features)

        X_6 = self.unpool6(global_features)
        X_6 = self.gconv6(self._get_decoder_input(X_4, X_6), self.A_4)

        X_7 = self.unpool7(X_6)
        X_7 = self.gconv7(self._get_decoder_input(X_3, X_7), self.A_3)

        X_8 = self.unpool8(X_7)
        X_8 = self.gconv8(self._get_decoder_input(X_2, X_8), self.A_2)

        X_9 = self.unpool9(X_8)
        X_9 = self.gconv9(self._get_decoder_input(X_1, X_9), self.A_1)

        X_10 = self.unpool10(X_9)
        X_10 = self.gconv10(self._get_decoder_input(X_0, X_10), self.A_0)

        return X_10


class GraphNetBatchNorm(torch.nn.Module):
    def __init__(self, in_features=2, out_features=2):
        super().__init__()

        self.A_hat = Parameter(torch.eye(29).float().cuda(), requires_grad=True)

        self.gconv1 = GraphConv(in_features, 128)
        self.gconv2 = GraphConv(128, 16)
        self.gconv3 = GraphConv(16, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        X_2 = self.gconv3(X_1, self.A_hat)

        return X_2
