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
import wandb
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
        self._lambda1 = 0.1
        self._lambda2 = 1
        self._epoch = 0
        self._exit = False
        signal.signal(signal.SIGINT, self._exit_gracefully)

    def _exit_gracefully(self, *args):
        self._exit = True

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
        use_scheduler: bool = True,
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

    def _testing_step(self, *args, **kargs):
        raise NotImplementedError("_testing_step() not implemented!")

    def _backup(self):
        print(f"-> Saving model to {self._checkpoint_path}...")
        torch.save(
            {
                "backup": True,
                "model_state_dict": self.model.state_dict(),
            },
            os.path.join(
                self._checkpoint_path,
                f"backup_weights.tar",
            ),
        )

    def _restore(self, opt, scheduler, resume_training: bool = True) -> float:
        print(f"[*] Restoring from checkpoint: {self._model_path}")
        checkpoint = torch.load(self._model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        val_loss = float("+inf")
        if resume_training and "backup" not in checkpoint.keys():
            self._epoch = checkpoint["epoch"] + 1
            opt.load_state_dict(checkpoint["opt_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            val_loss = checkpoint["val_loss"]
        return val_loss

    def _checkpoint(self, epoch, train_loss, val_loss, val_mae_loss, state_dicts):
        print(f"-> Saving model to {self._checkpoint_path}...")
        wandb.run.summary["best_val_mse"] = val_loss
        wandb.run.summary["best_val_mae"] = val_mae_loss
        torch.save(
            state_dicts,
            os.path.join(
                self._checkpoint_path,
                f"epoch_{epoch}_train_loss-{train_loss:.6f}_val_loss-{val_loss:.6f}.tar",
            ),
        )
