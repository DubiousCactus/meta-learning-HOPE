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
from util.utils import compute_curve, plot_curve
from typing import List, Union, Optional
from algorithm.base import BaseTrainer
from collections import namedtuple
from functools import partial
from tqdm import tqdm

import logging
import wandb
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

    def train(
        self,
        batch_size: int = 32,
        iterations: int = 1000,
        fast_lr: float = 0.01,
        meta_lr: float = None,
        optimizer: str = "adam",
        val_every: int = 100,
        resume: bool = True,
        use_scheduler: bool = True,
    ):
        wandb.watch(self.model)
        if optimizer == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=fast_lr)
        else:
            raise ValueError(f"{optimizer} is not a valid outer optimizer")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=iterations,
            eta_min=0.00001,
            last_epoch=self._epoch - 1,
            verbose=True,
        )
        past_val_loss = float("+inf")
        if self._model_path:
            past_val_loss = self._restore(opt, scheduler, resume_training=resume)
        avg_val_loss, avg_val_mae_loss = 0.0, 0.0
        for epoch in range(self._epoch, iterations):
            self.model.train()
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
                val_losses, val_mae_losses = [], []
                print("Computing validation error...")
                for batch in tqdm(self.dataset.val, dynamic_ncols=True):
                    if self._exit:
                        self._backup()
                        return
                    val_losses.append(self._testing_step(batch))
                    val_mae_losses.append(self._testing_step(batch, compute="mae"))
                avg_val_loss = float(torch.Tensor(val_losses).mean().item())
                avg_val_mae_loss = float(torch.Tensor(val_mae_losses).mean().item())

            avg_train_loss = torch.Tensor(train_losses).mean().item()
            wandb.log({"train_loss": avg_train_loss}, step=epoch)
            print(f"==========[Epoch {epoch}]==========")
            print(f"Training Loss: {avg_train_loss:.6f}")
            if (epoch + 1) % val_every == 0:
                print(f"Validation Loss: {avg_val_loss:.6f}")
                print(f"Validation MAE Loss: {avg_val_mae_loss:.6f}")
                wandb.log(
                    {"val_loss": avg_val_loss, "val_mae_loss": avg_val_mae_loss},
                    step=epoch,
                )
            print("============================================")
            # Model checkpointing
            if (epoch + 1) % val_every == 0 and avg_val_loss < past_val_loss:
                state_dicts = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val_loss,
                }
                self._checkpoint(
                    epoch, avg_train_loss, avg_val_loss, avg_val_mae_loss, state_dicts
                )
                past_val_loss = avg_val_loss
            if use_scheduler:
                scheduler.step()

    def test(
        self,
        batch_size: int = 32,
        runs: int = 1, # Unused
        fast_lr: float = 0.01,
        meta_lr: float = None,
        visualize: bool = False,
        plot: bool = False,
    ):
        if not self._model_path:
            print(f"[!] Testing a randomly initialized model!")
        else:
            print(f"[*] Restoring from checkpoint: {self._model_path}")
            checkpoint = torch.load(self._model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        MPJPEs, MPCPEs = [], []
        PJPEs, PCPEs = [], []
        thresholds = torch.linspace(10, 100, (100-10)//5 + 1)

        for batch in tqdm(self.dataset.test, dynamic_ncols=True):
            if self._exit:
                return
            res = self._testing_step(batch, compute=["pjpe"])
            PJPEs.append(res["pjpe"])
            MPJPEs.append(res["pjpe"].mean())
            if visualize:
                self._testing_step_vis(batch)

        print("-> Computing PCK curves...")
        # Compute the PCK curves (hand joints)
        auc_pck, pck = compute_curve(PJPEs, thresholds, 21)
        if plot:
            plot_curve(pck, thresholds, "baseline_pck.png")
        mpjpe = float(torch.Tensor(MPJPEs).mean().item())
        print(f"\n\n==========[Test Error]==========")
        print(f"Mean Per Joint Pose Error: {mpjpe:.6f}")
        print(f"Area Under Curve for PCK: {auc_pck:.6f}")
