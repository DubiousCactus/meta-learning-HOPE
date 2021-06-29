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
        model_path: str = None,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
    ):
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number
        self._model_path = model_path
        self.model = select_model(model_name)
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_number)
        self.dataset = dataset
        self._checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
        if not os.path.isdir(self._checkpoint_path):
            os.makedirs(self._checkpoint_path)
        self.inner_criterion = torch.nn.MSELoss()
        self._lambda1 = 0.01
        self._lambda2 = 1
        self._epoch = 0

    def train(
        self,
        meta_batch_size: int = 16,
        iterations: int = 1000,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        lr_step: int = 100,
        lr_step_gamma: float = 0.5,
        save_every: int = 100,
        val_every: int = 100,
        resume: bool = True,
    ):
        raise NotImplementedError

    def test(
        self,
        meta_batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
    ):
        raise NotImplementedError
