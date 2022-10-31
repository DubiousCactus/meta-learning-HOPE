#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Regular training wrappers
"""

from util.utils import plot_3D_pred_gt, select_cnn_model
from data.dataset.base import BaseDatasetTaskLoader
from algorithm.regular import RegularTrainer

from typing import List

import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch


class Regular_CNNTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        model_path: str = None,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            select_cnn_model(cnn_def, hand_only),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        joints, _ = self.model(inputs)
        loss = self.inner_criterion(joints, labels3d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            joints, _ = self.model(inputs)
            res = None
            if type(compute) is str:
                """
                This will be used when validating.
                """
                if compute == "mse":
                    res = self.inner_criterion(joints, labels3d).detach()
                elif compute == "mae":
                    res = F.l1_loss(joints, labels3d).detach()
                elif compute == "mpjpe":
                    # Hand-pose only
                    # Batched vector norm for row-wise elements
                    return (
                        torch.linalg.norm(
                            joints[:, :self._dim, :] - labels3d[:, :self._dim, :], dim=2
                        )
                        .detach()
                        .mean()
                    )
            elif type(compute) is list:
                """
                This will be used when testing.
                """
                res = {}
                for metric in compute:
                    if metric == "mse":
                        res[metric] = (self.inner_criterion(joints, labels3d).detach())
                    elif metric == "mae":
                        res[metric] = (F.l1_loss(joints, labels3d).detach())
                    elif metric == "pjpe":
                        # Hand-pose only
                        # Batched vector norm for row-wise elements
                        res[metric] = (
                            torch.linalg.norm(
                                joints[:, :21, :] - labels3d[:, :21, :], dim=2
                            )
                            .detach()
                        )
                    elif metric == "pcpe":
                        # Object-pose only
                        # Batched vector norm for row-wise elements
                        res[metric] = (
                            torch.linalg.norm(
                                joints[:, 21:, :] - labels3d[:, 21:, :], dim=2
                            )
                            .detach()
                        )
        assert res is not None, f"{compute} is not a valid metric!"
        return res

    def _testing_step_vis(self, batch: tuple):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            joints, _ = self.model(inputs)
            mean, std = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32
            ), torch.tensor([0.221, 0.224, 0.225], dtype=torch.float32)
            unnormalize = transforms.Normalize(
                mean=(-mean / std).tolist(), std=(1.0 / std).tolist()
            )
            unnormalized_img = unnormalize(inputs[0])
            npimg = (
                (unnormalized_img * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )
            plot_3D_pred_gt(joints[0].cpu(), npimg, labels3d[0].cpu())
