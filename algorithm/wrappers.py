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

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer, MetaBatch
from algorithm.regular import RegularTrainer
from HOPE.models.graphunet import GraphNet
from HOPE.utils.model import select_model
from typing import List


import torch


class MAML_HOPETrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_querries,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "hopenet",
            dataset,
            checkpoint_path,
            k_shots,
            n_querries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: MetaBatch, learner):
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
        e_loss2d_init = self.inner_criterion(e_outputs2d_init, q_labels2d)
        e_loss2d = self.inner_criterion(e_outputs2d, q_labels2d)
        e_loss3d = self.inner_criterion(e_outputs3d, q_labels3d)
        query_loss = (
            (self._lambda1) * e_loss2d_init
            + (self._lambda1) * e_loss2d
            + (self._lambda2) * e_loss3d
        )
        return query_loss


class MAML_ResnetTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_querries,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "resnet10",
            dataset,
            checkpoint_path,
            k_shots,
            n_querries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: MetaBatch, learner):
        s_inputs, s_labels2d, _ = batch.support
        q_inputs, q_labels2d, _ = batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs2d_init, _ = learner(s_inputs)
            support_loss = self.inner_criterion(outputs2d_init, s_labels2d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, _ = learner(q_inputs)
        query_loss = self.inner_criterion(e_outputs2d_init, q_labels2d)
        return query_loss


class MAML_GraphUNetTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_querries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "graphunet",
            dataset,
            checkpoint_path,
            k_shots,
            n_querries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: MetaBatch, learner):
        _, s_labels2d, s_labels3d = batch.support
        _, q_labels2d, q_labels3d = batch.query
        if self._use_cuda:
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs3d = learner(s_labels2d)
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        outputs3d = learner(q_labels2d)
        query_loss = self.inner_criterion(outputs3d, q_labels3d)
        return query_loss


class Regular_ResnetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "resnet10",
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple, backward: bool = True):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        outputs2d_init, _ = self.model(inputs)
        loss = self.inner_criterion(outputs2d_init, labels2d)
        if backward:
            loss.backward()
        return loss


class Regular_GraphUNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            "graphunet",
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple, backward: bool = True):
        _, labels2d, labels3d = batch
        if self._use_cuda:
            labels2d = labels2d.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        outputs3d = self.model(labels2d)
        loss = self.inner_criterion(outputs3d, labels3d)
        if backward:
            loss.backward()
        return loss


class Regular_GraphNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        resnet_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphNet(
                in_features=514, out_features=2
            ),  # 514 for the output features of resnet10
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self._resnet = select_model("resnet10")
        if use_cuda and torch.cuda.is_available():
            self._resnet = self._resnet.cuda()
            self._resnet = torch.nn.DataParallel(self._resnet, device_ids=gpu_numbers)
        if resnet_path:
            print(f"[*] Loading ResNet state dict form {resnet_path}")
            ckpt = torch.load(resnet_path)
            self._resnet.load_state_dict(ckpt["model_state_dict"])
        else:
            print("[!] ResNet is randomly initialized!")
        self._resnet.eval()

    def _training_step(self, batch: tuple, backward: bool = True):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            points2D_init, features = self._resnet(inputs)
            features = features.unsqueeze(1).repeat(1, 29, 1)
            in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.model(in_features)
        loss = self.inner_criterion(points2D, labels2d)
        if backward:
            loss.backward()
        return loss
