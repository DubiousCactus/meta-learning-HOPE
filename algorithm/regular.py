#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Typical one-objective training.
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.base import BaseTrainer
from collections import namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import learn2learn as l2l
import torch
import os


class RegularTrainer(BaseTrainer):
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
        super().__init__(
            model_name,
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _restore(self, opt, scheduler, resume_training: bool = True):
        checkpoint = torch.load(self._model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        if resume_training:
            self._epoch = checkpoint["epoch"] + 1
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            return checkpoint["val_loss"]


    def train(
        self,
        batch_size: int = 32,
        iterations: int = 1000,
        fast_lr: float = 0.01,
        meta_lr: float = None,
        lr_step: int = 100,
        lr_step_gamma: float = 0.5,
        val_every: int = 100,
        resume: bool = True,
    ):
        opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_step_gamma, verbose=True
        )
        scheduler.last_epoch = self._epoch
        past_val_loss = float("+inf")
        if self._model_path:
            past_val_loss = self._restore(opt, scheduler, resume_training=resume)
        for epoch in range(self._epoch, iterations):
            train_losses = []
            for batch in tqdm(self.dataset.train):
                opt.zero_grad()
                loss = self._training_step(batch)
                train_losses.append(loss.detach())
                opt.step()

            if (epoch + 1) % val_every == 0:
                val_losses = []
                print("Computing validation error...")
                for batch in tqdm(self.dataset.val):
                    val_losses.append(self._training_step(batch, backward=False).detach())
                avg_val_loss = torch.Tensor(val_losses).mean().item()

            avg_train_loss = torch.Tensor(train_losses).mean().item()
            print(f"==========[Epoch {epoch}]==========")
            print(f"Training Loss: {avg_train_loss:.6f}")
            if (epoch + 1) % val_every == 0:
                print(f"Validation Loss: {avg_val_loss:.6f}")
            print("============================================")
            scheduler.step()
            # Model checkpointing
            if (epoch + 1) % val_every == 0 and avg_val_loss < past_val_loss:
                print(f"-> Saving model to {self._checkpoint_path}...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    os.path.join(
                        self._checkpoint_path,
                        f"epoch_{epoch}_train_loss-{avg_train_loss:.6f}_val_loss-{avg_val_loss:.6f}.tar",
                    ),
                )
                past_val_loss = avg_val_loss


    def test(
        self,
        batch_size: int = 32,
        fast_lr: float = 0.01,
        meta_lr: float = None,
    ):
        raise NotImplementedError