#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
MAML training wrappers
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer, MetaBatch
from util.utils import select_cnn_model

from typing import List, Optional

import torch.nn.functional as F
import torch

class MAML_CNNTrainer(MAMLTrainer):
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
            task_aug=task_aug,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(
        self, batch: MetaBatch, learner, epoch=None, compute="mse", msl=True
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

        if self._task_aug == "permute":
            # Apply the same random permutation of target vector dims
            dims = s_labels3d[0].shape[0] # Permute the joints, not the axes
            perms = torch.randperm(dims)
            s_labels3d = s_labels3d[:, perms, :]
            q_labels3d = q_labels3d[:, perms, :]

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            joints, _ = learner(s_inputs)
            support_loss = self.inner_criterion(joints, s_labels3d)
            learner.adapt(support_loss, epoch=epoch)
            if msl:  # Multi-step loss
                q_joints, _ = learner(q_inputs)
                query_loss += self._step_weights[step] * criterion(
                    q_joints, q_labels3d
                )

        # Evaluate the adapted model on the query set
        if not msl:
            q_joints, _ = learner(q_inputs)
            query_loss = criterion(q_joints, q_labels3d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, epoch=None, compute="mse"
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, _, s_labels3d = meta_batch.support
        q_inputs, _, q_labels3d = meta_batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            joints, _ = learner(s_inputs)
            support_loss = self.inner_criterion(joints, s_labels3d)
            learner.adapt(support_loss, epoch=epoch)

        with torch.no_grad():
            q_joints, _ = learner(q_inputs)
        return criterion(q_joints, q_labels3d)
