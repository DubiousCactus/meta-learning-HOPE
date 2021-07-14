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
from typing import List, Union
from functools import partial
from tqdm import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt
import learn2learn as l2l
import logging
import torch
import os


class RegularTrainer(BaseTrainer):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            model,
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
        if resume_training and "backup" not in checkpoint.keys():
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
        shown = False
        if self._model_path:
            saved_val_loss = self._restore(opt, scheduler, resume_training=resume)
            if resume:
                past_val_loss = saved_val_loss
                shown = True
        if not shown:
            log.info(f"=====================================")
            log.info(f"fast_lr={fast_lr} - batch_size={batch_size}")
            log.info(f"=====================================")
        avg_val_loss = 0.0
        for epoch in range(self._epoch, iterations):
            self.model.train()
            scheduler.step()
            train_losses = []
            for batch in tqdm(self.dataset.train, dynamic_ncols=True):
                if self._exit:
                    self._backup()
                    return
                opt.zero_grad()
                loss = self._training_step(batch)
                train_losses.append(loss)
                opt.step()

            if (epoch + 1) % val_every == 0:
                self.model.eval()
                val_losses = []
                print("Computing validation error...")
                for batch in tqdm(self.dataset.val, dynamic_ncols=True):
                    val_losses.append(self._testing_step(batch))
                avg_val_loss = torch.Tensor(val_losses).mean().item()

            avg_train_loss = torch.Tensor(train_losses).mean().item()
            log.info(f"[Epoch {epoch}]: Training Loss: {avg_train_loss:.6f}")
            print(f"==========[Epoch {epoch}]==========")
            print(f"Training Loss: {avg_train_loss:.6f}")
            if (epoch + 1) % val_every == 0:
                print(f"Validation Loss: {avg_val_loss:.6f}")
                log.info(f"[Epoch {epoch}]: Validation Loss: {avg_val_loss:.6f}")
            print("============================================")
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
        avg_mse_loss, avg_mae_loss, mse_losses, mae_losses = 0.0, 0.0, [], []
        for batch in tqdm(self.dataset.test, dynamic_ncols=True):
            mae_losses.append(self._testing_step(batch, loss_fn=F.l1_loss))
            mse_losses.append(self._testing_step(batch))
        avg_mse_loss = torch.Tensor(mse_losses).mean().item()
        avg_mae_loss = torch.Tensor(mae_losses).mean().item()
        print(f"[*] Average MSE test loss: {avg_mse_loss:.6f}")
        print(f"[*] Average MAE test loss: {avg_mae_loss:.6f}")

        # Percentage of Correct Keypoints (3D-PCK for hand) / Poses (3D-PCP for object) merged:
        for thresh in range(0, 80, 5):
            correct_poses = 0
            for batch in tqdm(self.dataset.test, dynamic_ncols=True):
                mean_norms = torch.mean(
                    self._testing_step(
                        batch,
                        loss_fn=lambda pred, target: torch.norm((pred - target), dim=2),
                    ),
                    dim=1,
                )
                correct_poses += (
                    torch.where(mean_norms < thresh, 1, 0)
                    .count_nonzero()
                    .item()
                )
            print(
                f"[*] Percentage of Correct Poses (PCP-3D) with threshold={thresh}: {correct_poses/len(self.dataset.test):.6f}"
            )
