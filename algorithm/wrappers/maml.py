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
from model.hopenet import HOPENet

from typing import List

import torch.nn.functional as F
import torch


class MAML_HOPETrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries,
        inner_steps: int,
        cnn_def: str,
        resnet_path: str,
        graphnet_path: str,
        graphunet_path: str,
        model_path: str = None,
        first_order: bool = False,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
            dataset,
            checkpoint_path,
            k_shots,
            n_queries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(
        self, batch: MetaBatch, learner, compute="mse"
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, s_labels2d, s_labels3d = batch.support
        q_inputs, q_labels2d, q_labels3d = batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs2d_init, outputs2d, outputs3d = learner(s_inputs)
            loss2d_init = self.inner_criterion(outputs2d_init, s_labels2d)
            loss2d = self.inner_criterion(outputs2d, s_labels2d)
            loss3d = self.inner_criterion(outputs3d, s_labels3d)
            support_loss = (
                (self._lambda1) * loss2d_init
                + (self._lambda1) * loss2d
                + (self._lambda2) * loss3d
            )
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, e_outputs2d, e_outputs3d = learner(q_inputs)
        e_loss2d_init = criterion(e_outputs2d_init, q_labels2d)
        e_loss2d = criterion(e_outputs2d, q_labels2d)
        e_loss3d = criterion(e_outputs3d, q_labels3d)
        query_loss = (
            (self._lambda1) * e_loss2d_init
            + (self._lambda1) * e_loss2d
            + (self._lambda2) * e_loss3d
        )
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, compute="mse"
    ):
        return self._training_step(meta_batch, learner, compute)


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
        task_aug: bool = True,
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

        if self._task_aug:
            # Apply the same random permutation of target vector dims
            dims = s_labels3d[0].shape[0] # Permute the joints, not the axes
            perms = torch.randperm(dims)
            s_labels3d = s_labels3d[:, perms, :]
            q_labels3d = q_labels3d[:, perms, :]

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            joints, _ = learner(s_inputs)
            joints -= joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1) # Root alignment
            support_loss = self.inner_criterion(joints, s_labels3d)
            learner.adapt(support_loss, epoch=epoch)
            if msl:  # Multi-step loss
                q_joints, _ = learner(q_inputs)
                q_joints -= q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1) # Root alignment
                query_loss += self._step_weights[step] * criterion(
                    q_joints, q_labels3d
                )

        # Evaluate the adapted model on the query set
        if not msl:
            q_joints, _ = learner(q_inputs)
            q_joints -= q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1) # Root alignment
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
            joints -= joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1) # Root alignment
            support_loss = self.inner_criterion(joints, s_labels3d)
            learner.adapt(support_loss, epoch=epoch)

        with torch.no_grad():
            q_joints, _ = learner(q_inputs)
            q_joints -= q_joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1) # Root alignment
        return criterion(q_joints, q_labels3d)


class MAML_GraphUNetTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        multi_step_loss: bool = True,
        msl_num_epochs: int = 1000,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "graphunet",
            dataset,
            checkpoint_path,
            k_shots,
            n_queries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            multi_step_loss=multi_step_loss,
            msl_num_epochs=msl_num_epochs,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(
        self, batch: MetaBatch, learner, compute="mse", msl=True
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        _, s_labels2d, s_labels3d = batch.support
        _, q_labels2d, q_labels3d = batch.query
        query_loss = 0.0
        if self._use_cuda:
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # with torch.no_grad():
        # avg_norm = []
        # for p in learner.parameters():
        # avg_norm.append(torch.linalg.norm(p.data))
        # print(torch.tensor(avg_norm))
        # avg_norm = torch.tensor(avg_norm).mean().item()
        #     print(f"Average inner weight norm: {avg_norm:.2f}")

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            outputs3d = learner(s_labels2d)
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            learner.adapt(support_loss)
            if msl:  # Multi-step loss
                q_outputs3d = learner(q_labels2d)
                query_loss += self._step_weights[step] * criterion(
                    q_outputs3d, q_labels3d
                )

        # with torch.no_grad():
        # avg_norm = []
        # for p in learner.parameters():
        # avg_norm.append(torch.linalg.norm(p.data))
        # print(torch.tensor(avg_norm))
        # avg_norm = torch.tensor(avg_norm).mean().item()
        # print(f"Average inner weight norm: {avg_norm:.2f}")

        # Evaluate the adapted model on the query set
        if not msl:
            q_outputs3d = learner(q_labels2d)
            query_loss = criterion(q_outputs3d, q_labels3d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, compute="mse"
    ):
        return self._training_step(
            meta_batch, learner, compute, msl=False
        )
