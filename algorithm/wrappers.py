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

import torch.optim as optim
import torch.nn as nn
import torch

from data.dataset import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer
from data.utils import DatasetFactory

from torch.autograd import Variable
from abc import ABC


class HOPETrainer(MAMLTrainer):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        batch_size: int,
        k_shots: int,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "hopenet",
            dataset_name,
            dataset_root,
            batch_size,
            k_shots,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        inputs, labels2d, labels3d = batch
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        # TODO: Do this in the construction of the tasks dataset
        if self._use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=self._gpu_number[0])
            labels2d = labels2d.float().cuda(device=self._gpu_number[0])
            labels3d = labels3d.float().cuda(device=self._gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs2d_init, outputs2d, outputs3d = learner(inputs)
            loss2d_init = self.inner_criterion(outputs2d_init, labels2d)
            loss2d = self.inner_criterion(outputs2d, labels2d)
            loss3d = self.inner_criterion(outputs3d, labels3d)
            support_loss = (
                (self._lambda1) * loss2d_init
                + (self._lambda1) * loss2d
                + (self._lambda2) * loss3d
            )
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, e_outputs2d, e_outputs3d = learner(inputs)
        e_loss2d_init = self.inner_criterion(e_outputs2d_init, labels2d)
        e_loss2d = self.inner_criterion(e_outputs2d, labels2d)
        e_loss3d = self.inner_criterion(e_outputs3d, labels3d)
        query_loss = (
            (self._lambda1) * e_loss2d_init
            + (self._lambda1) * e_loss2d
            + (self._lambda2) * e_loss3d
        )
        return query_loss


class ResnetTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        batch_size: int,
        k_shots: int,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "resnet10",
            dataset_name,
            dataset_root,
            batch_size,
            k_shots,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        inputs, labels2d, _ = batch
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)

        # TODO: Do this in the construction of the tasks dataset
        if self._use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=self._gpu_number[0])
            labels2d = labels2d.float().cuda(device=self._gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs2d_init, _ = learner(inputs)
            support_loss = self.inner_criterion(outputs2d_init, labels2d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, _ = learner(inputs)
        query_loss = self.inner_criterion(e_outputs2d_init, labels2d)
        return query_loss


class GraphUNetTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        batch_size: int,
        k_shots: int,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "graphunet",
            dataset_name,
            dataset_root,
            batch_size,
            k_shots,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        _, labels2d, labels3d = batch
        # wrap them in Variable
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        # TODO: Do this in the construction of the tasks dataset
        if self._use_cuda and torch.cuda.is_available():
            labels2d = labels2d.float().cuda(device=self._gpu_number[0])
            labels3d = labels3d.float().cuda(device=self._gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs3d = learner(labels2d)
            support_loss = self.inner_criterion(outputs3d, labels3d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        outputs3d = learner(labels2d)
        query_loss = self.inner_criterion(outputs3d, labels3d)
        return query_loss
