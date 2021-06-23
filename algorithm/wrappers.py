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
import numpy as np
import torch

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer, MetaBatch

from abc import ABC


# TODO: Pass in a simple function to MAML instead of this useless inheritance!
class MAML_HOPETrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        k_shots: int,
        n_querries,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "hopenet",
            dataset,
            k_shots,
            n_querries,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(
            self, batch: MetaBatch, learner, steps: int, shots: int
    ):
        s_inputs, s_labels2d, s_labels3d = batch.support
        q_inputs, q_labels2d, q_labels3d = batch.query

        # Adapt the model on the support set
        for step in range(steps):
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
        k_shots: int,
        n_querries,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "resnet10",
            dataset,
            k_shots,
            n_querries,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(
            self, batch: MetaBatch, learner, steps: int, shots: int
    ):
        s_inputs, s_labels2d, _ = batch.support
        q_inputs, q_labels2d, _ = batch.query

        # Adapt the model on the support set
        for step in range(steps):
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
        k_shots: int,
        n_querries: int,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        super().__init__(
            "graphunet",
            dataset,
            k_shots,
            n_querries,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(
            self, batch: MetaBatch, learner, steps: int, shots: int
    ):
        _, s_labels2d, s_labels3d = batch.support
        _, q_labels2d, q_labels3d = batch.query

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs3d = learner(s_labels2d)
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        outputs3d = learner(q_labels2d)
        query_loss = self.inner_criterion(outputs3d, q_labels3d)
        return query_loss
