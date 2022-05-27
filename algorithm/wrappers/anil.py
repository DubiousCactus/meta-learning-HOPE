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

from typing import List, Optional

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
        beta: float = 1e-7,
        reg_bottleneck_dim: int = 512,
        meta_reg: bool = True,
        task_aug: Optional[str] = None,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            select_cnn_model(cnn_def, hand_only),
            dataset,
            checkpoint_path,
            k_shots,
            n_queries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            multi_step_loss=multi_step_loss,
            msl_num_epochs=msl_num_epochs,
            beta=beta,
            reg_bottleneck_dim=reg_bottleneck_dim,
            meta_reg=meta_reg,
            task_aug=task_aug,
            hand_only=hand_only,
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

        if self._task_aug not in ["permute", "discrete_noise", "shift", None]:
            raise KeyError("Parameter task_aug must be one of [None, shift, permute, discrete_noise]")

        if self._task_aug == "permute":
            # Apply the same random permutation of target vector dims
            dims = s_labels3d.shape[1] # Permute the joints, not the axes
            perms = torch.randperm(dims)
            s_labels3d = s_labels3d[:, perms, :]
            q_labels3d = q_labels3d[:, perms, :]
        elif self._task_aug == "discrete_noise":
            # Add a noise value sampled form a discrete set to a randomly sampled axis
            noise_values = np.linspace(0, 1, self._task_aug_noise_values+1)[:-1] # Ignore 1 but include 0
            noise = np.random.choice(noise_values)
            axis = np.random.randint(0, 3)
            # My intuitiion is that it'd make more sense to ignore the root joint, which the
            # network learns to always be 0. So adding noise to it might interfere in the
            # learning and cause the gradients to be noisier.
            s_labels3d[:, 1:, axis] += noise
            q_labels3d[:, 1:, axis] += noise
        elif self._task_aug == "shift":
            # Shift the vertices/joints, such that the order is kept and the gradients are less
            # noisy.
            dims = s_labels3d.shape[1] # Permute the joints, not the axes
            n_shifts = torch.randint(dims, (1,)).item()
            s_labels3d = torch.roll(s_labels3d, n_shifts, 1)
            q_labels3d = torch.roll(q_labels3d, n_shifts, 1)

        s_inputs_features = features(s_inputs)
        q_inputs_features = features(q_inputs)
        if self._meta_reg:
            # Encoding of inputs through BBB for Meta-Regularisation
            s_inputs_features, _ = self.encoder(s_inputs_features)
            q_inputs_features, kl = self.encoder(q_inputs_features)

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            joints = head(s_inputs_features).view(-1, self._dim, 3)
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss, epoch=epoch)
            if msl:  # Multi-step loss
                q_joints = head(q_inputs_features).view(-1, self._dim, 3)
                query_loss += self._step_weights[step] * criterion(q_joints, q_labels3d)

        del s_inputs
        # Evaluate the adapted model on the query set
        if not msl:
            q_joints = head(q_inputs_features).view(-1, self._dim, 3)
            query_loss = criterion(q_joints, q_labels3d)

        if self._meta_reg:
            # Only add the KL divergence term once, since it's the same value per query set
            query_loss += self._beta * kl
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
            s_inputs_features = features(s_inputs)
            q_inputs_features = features(q_inputs)
            if self._meta_reg:
                # Encoding of inputs through BBB for Meta-Regularisation
                s_inputs_features, _ = self.encoder(s_inputs_features)
                q_inputs_features, _ = self.encoder(q_inputs_features)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            joints = head(s_inputs_features).view(-1, self._dim, 3)
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss, epoch=epoch)

        with torch.no_grad():
            q_joints = head(q_inputs_features).view(-1, self._dim, 3)

        res = None
        if type(compute) is str:
            """
            This will be used when validating.
            """
            if compute == "mse":
                res = self.inner_criterion(q_joints, q_labels3d).detach()
            elif compute == "mae":
                res = F.l1_loss(q_joints, q_labels3d).detach()
            elif compute == "mpjpe":
                # Hand-pose only
                # Batched vector norm for row-wise elements
                return (
                    torch.linalg.norm(
                        q_joints[:, :self._dim, :] - q_labels3d[:, :self._dim, :], dim=2
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
                    res[metric] = (self.inner_criterion(q_joints, q_labels3d).detach())
                elif metric == "mae":
                    res[metric] = F.l1_loss(q_joints, q_labels3d).detach()
                elif metric == "pjpe":
                    # Hand-pose only
                    # Batched vector norm for row-wise elements
                    res[metric] = (
                        torch.linalg.norm(
                            q_joints[:, :self._dim, :] - q_labels3d[:, :self._dim, :], dim=2
                        )
                        .detach()
                    )
                elif metric == "pcpe":
                    # Object-pose only
                    # Batched vector norm for row-wise elements
                    res[metric] = (
                        torch.linalg.norm(
                            q_joints[:, self._dim:, :] - q_labels3d[:, self._dim:, :], dim=2
                        )
                        .detach()
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
            joints = head(s_inputs).view(-1, self._dim, 3)
            support_loss = self.inner_criterion(joints, s_labels3d)
            head.adapt(support_loss)

        with torch.no_grad():
            q_inputs_f = features(q_inputs)
            q_joints = head(q_inputs_f).view(-1, self._dim, 3)
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
            # plot_3D_pred_gt(q_joints[0].cpu(), npimg, q_labels3d[0].cpu())
