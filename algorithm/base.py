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
from typing import List, Union
from abc import ABC

import signal
import torch
import os


class BaseTrainer(ABC):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        self._use_cuda = use_cuda
        self._gpu_number = gpu_numbers[0]
        self._model_path = model_path
        if type(model) is str:
            self.model: torch.nn.Module = select_model(model)
        else:
            self.model: torch.nn.Module = model
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_numbers)
        self.dataset = dataset
        self._checkpoint_path = checkpoint_path
        if not os.path.isdir(self._checkpoint_path):
            os.makedirs(self._checkpoint_path)
        self.inner_criterion = torch.nn.MSELoss(reduction="mean")
        self._lambda1 = 0.01
        self._lambda2 = 1
        self._epoch = 0
        self._exit = False
        signal.signal(signal.SIGINT, self._exit_gracefully)

    def _exit_gracefully(self, *args):
        raise NotImplementedError

    def train(
        self,
        batch_size: int = 16,
        iterations: int = 1000,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        lr_step: int = 100,
        lr_step_gamma: float = 0.5,
        val_every: int = 100,
        resume: bool = True,
    ):
        raise NotImplementedError

    def test(
        self,
        batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
    ):
        raise NotImplementedError

    def _training_step(self, *args, **kargs):
        raise NotImplementedError("_training_step() not implemented!")
