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
from typing import List
from tqdm import tqdm

import matplotlib.pyplot as plt
import learn2learn as l2l
import logging
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
        gpu_numbers: List = [0],
    ):
        super().__init__(
            model_name,
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _restore(self, opt, scheduler, resume_training: bool = True):
        print(f"[*] Restoring from checkpoint: {self._model_path}")
        checkpoint = torch.load(self._model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if resume_training and 'backup' not in checkpoint.keys():
            self._epoch = checkpoint["epoch"] + 1
            opt.load_state_dict(checkpoint["opt_state_dict"])
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
        log = logging.getLogger(__name__)
        opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_step_gamma, verbose=True
        )
        scheduler.last_epoch = self._epoch
        past_val_loss = float("+inf")
        if self._model_path:
            saved_val_loss = self._restore(opt, scheduler, resume_training=resume)
            if resume:
                past_val_loss = saved_val_loss
        else:
            log.info(f"=====================================")
            log.info(f"fast_lr={fast_lr} - batch_size={batch_size}")
            log.info(f"=====================================")
        avg_val_loss = .0
        for epoch in range(self._epoch, iterations):
            self.model.train()
            train_losses = []
            for batch in tqdm(self.dataset.train):
                if self._exit:
                    self._backup()
                    return
                opt.zero_grad()
                loss = self._training_step(batch)
                train_losses.append(loss.detach())
                opt.step()

            if (epoch + 1) % val_every == 0:
                self.model.eval()
                val_losses = []
                print("Computing validation error...")
                for batch in tqdm(self.dataset.val):
                    val_losses.append(
                        self._training_step(batch, backward=False).detach()
                    )
                avg_val_loss = torch.Tensor(val_losses).mean().item()

            avg_train_loss = torch.Tensor(train_losses).mean().item()
            log.info(f"[Epoch {epoch}]: Training Loss: {avg_train_loss:.6f}")
            print(f"==========[Epoch {epoch}]==========")
            print(f"Training Loss: {avg_train_loss:.6f}")
            if (epoch + 1) % val_every == 0:
                print(f"Validation Loss: {avg_val_loss:.6f}")
                log.info(f"[Epoch {epoch}]: Validation Loss: {avg_val_loss:.6f}")
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

    def _exit_gracefully(self, *args):
        self._exit = True

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

    def test(
        self,
        batch_size: int = 32,
        fast_lr: float = 0.01,
        meta_lr: float = None,
    ):
        if not self._model_path:
            print(f"[!] Testing a randomly initialized model!")
        else:
            print(f"[*] Restoring from checkpoint: {self._model_path}")
            checkpoint = torch.load(self._model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        avg_loss, losses = .0, []
        with torch.no_grad():
            for batch in tqdm(self.dataset.test):
                losses.append(
                    self._training_step(batch, backward=False).detach()
                )
            avg_loss = torch.Tensor(losses).mean().item()
        print(f"[*] Average test loss: {avg_loss:.6f}")
