#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Part-specific training wrappers.
"""

from util.utils import plot_3D_pred_gt, select_cnn_model
from data.dataset.base import BaseDatasetTaskLoader
from algorithm.anil import ANILTrainer
from algorithm.maml import MetaBatch

from typing import List

import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch


class ANIL_CNNTrainer(ANILTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries,
        inner_steps: int,
        cnn_def: str,
        model_path: str = None,
        first_order: bool = False,
        multi_step_loss: bool = True,
        msl_num_epochs: int = 1000,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            select_cnn_model(cnn_def),
            dataset,
            checkpoint_path,
            k_shots,
            n_queries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            multi_step_loss=multi_step_loss,
            msl_num_epochs=msl_num_epochs,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(
        self,
        batch: MetaBatch,
        head,
        features,
        epoch,
        compute="mse",
        msl=True,
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, _, s_labels3d = batch.support
        q_inputs, _, q_labels3d = batch.query
        query_loss = 0.0
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        s_inputs = features(s_inputs)
        q_inputs_features = features(q_inputs)
        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            joints = head(s_inputs).view(-1, 29, 3)
            joints -= (
                joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss, epoch=epoch)
            if msl:  # Multi-step loss
                q_joints = head(q_inputs_features).view(-1, 29, 3)
                q_joints -= (
                    q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
                )  # Root alignment
                query_loss += self._step_weights[step] * criterion(q_joints, q_labels3d)

        del s_inputs
        # Evaluate the adapted model on the query set
        if not msl:
            q_joints = head(q_inputs_features).view(-1, 29, 3)
            q_joints -= (
                q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment
            query_loss = criterion(q_joints, q_labels3d)
        return query_loss

    def _testing_step(
        self,
        meta_batch: MetaBatch,
        head,
        features,
        epoch=None,
        compute="mse",
    ):
        s_inputs, _, s_labels3d = meta_batch.support
        q_inputs, _, q_labels3d = meta_batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        with torch.no_grad():
            s_inputs = features(s_inputs)
        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            joints = head(s_inputs).view(-1, 29, 3)
            joints -= (
                joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss, epoch=epoch)

        with torch.no_grad():
            q_inputs = features(q_inputs)
            q_joints = head(q_inputs).view(-1, 29, 3)
            q_joints -= (
                q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment

        res = None
        if type(compute) is str:
            """
            This will be used when validating.
            """
            res = self.inner_criterion(q_joints, q_labels3d)
        elif type(compute) is list:
            """
            This will be used when testing.
            """
            res = []
            for metric in compute:
                if metric == "mse":
                    res.append(self.inner_criterion(q_joints, q_labels3d).detach())
                elif metric == "mae":
                    res.append(F.l1_loss(q_joints, q_labels3d).detach())
                elif metric == "mpjpe":
                    # Hand-pose only
                    # Batched vector norm for row-wise elements
                    res.append(
                        torch.linalg.norm(
                            q_joints[:, :21, :] - q_labels3d[:, :21, :], dim=2
                        )
                        .detach()
                        .mean()
                    )
                elif metric == "mpcpe":
                    # Object-pose only
                    # Batched vector norm for row-wise elements
                    res.append(
                        torch.linalg.norm(
                            q_joints[:, 21:, :] - q_labels3d[:, 21:, :], dim=2
                        )
                        .detach()
                        .mean()
                    )
        assert res is not None, f"{compute} is not a valid metric!"
        return res

    def _testing_step_vis(
        self,
        meta_batch: MetaBatch,
        head,
        features,
    ):
        s_inputs, _, s_labels3d = meta_batch.support
        q_inputs, _, q_labels3d = meta_batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        s_inputs = features(s_inputs)
        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            joints = head(s_inputs).view(-1, 29, 3)
            joints -= (
                joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss)

        with torch.no_grad():
            q_inputs_f = features(q_inputs)
            q_joints = head(q_inputs_f).view(-1, 29, 3)
            q_joints -= (
                q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
            )  # Root alignment
            mean, std = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32
            ), torch.tensor([0.221, 0.224, 0.225], dtype=torch.float32)
            unnormalize = transforms.Normalize(
                mean=(-mean / std).tolist(), std=(1.0 / std).tolist()
            )
            unnormalized_img = unnormalize(q_inputs[0])
            npimg = (
                (unnormalized_img * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )
            print(
                f"MSE={self.inner_criterion(q_joints, q_labels3d)} - MAE={F.l1_loss(q_joints, q_labels3d)}"
            )
            plot_3D_pred_gt(q_joints[0].cpu(), npimg, q_labels3d[0].cpu())
