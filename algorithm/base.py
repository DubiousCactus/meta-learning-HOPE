#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base training class.
"""

from data.dataset.base import BaseDatasetTaskLoader
from HOPE.utils.model import select_model
from abc import ABC

import torch
import os


class BaseTrainer(ABC):
    def __init__(
        self,
        model_name: str,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number
        self.model = select_model(model_name)
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_number)
        self.dataset = dataset
        self._checkpoint_path = checkpoint_path
        if not os.path.isdir(os.path.join(os.getcwd(), checkpoint_path)):
            os.makedirs(checkpoint_path)
        self.inner_criterion = torch.nn.MSELoss()
        # TODO: Add a scheduler in the meta-training loop?
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step,
        # gamma=lr_step_gamma)
        # self.scheduler.last_epoch = start
        self._lambda1 = 0.01
        self._lambda2 = 1

    def train(
        self,
        meta_batch_size: int = 16,
        iterations: int = 1000,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        steps: int = 1,
        shots: int = 10,
    ):
        raise NotImplementedError

    def test(
        self,
        meta_batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        steps: int = 1,
        shots: int = 10,
    ):
        raise NotImplementedError
