#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Almost No Inner-Loop meta-learning algorithm.
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer, MetaBatch
from model.cnn import ResNet, MobileNet
from collections import namedtuple
from typing import List, Union
from tqdm import tqdm

import matplotlib.pyplot as plt
import learn2learn as l2l
import logging
import torch
import wandb
import os


class ANILTrainer(MAMLTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        multi_step_loss: bool = True,
        msl_num_epochs: int = 1000,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            model,
            dataset,
            checkpoint_path,
            k_shots,
            n_queries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self.model: torch.nn.Module = model
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        self._step_weights = torch.ones(inner_steps) * (1.0 / inner_steps)
        self._msl = multi_step_loss
        self._msl_decay_rate = 1.0 / self._steps / msl_num_epochs
        self._msl_min_value_for_non_final_losses = torch.tensor(0.03 / self._steps)
        self._msl_max_value_for_final_loss = 1.0 - (
            (self._steps - 1) * self._msl_min_value_for_non_final_losses
        )

    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self._steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def train(
        self,
        batch_size: int = 32,
        iterations: int = 1000,
        fast_lr: float = 0.001,
        meta_lr: float = 0.01,
        lr_step: int = 100,
        lr_step_gamma: float = 0.5,
        max_grad_norm: float = 25.0,
        optimizer: str = "adam",
        val_every: int = 100,
        resume: bool = True,
        use_scheduler: bool = True,
    ):
        wandb.watch(self.model)
        maml = l2l.algorithms.MAML(
            self.model.head,
            lr=fast_lr,
            first_order=self._first_order,
            allow_unused=True,
        )
        all_parameters = list(self.model.features.parameters()) + list(
            maml.parameters()
        )
        if optimizer == "adam":
            opt = torch.optim.Adam(all_parameters, lr=meta_lr, amsgrad=False)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(all_parameters, lr=meta_lr)
        else:
            raise ValueError(f"{optimizer} is not a valid outer optimizer")

        iter_per_epoch = (
            len(self.dataset.train.dataset)
            // (batch_size * (self._k_shots + self._n_queries))
        ) + 1
        # From How to Train Your MAML:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=iterations * iter_per_epoch,
            eta_min=0.00001,
            last_epoch=self._epoch - 1,
        )
        past_val_loss = float("+inf")
        if self._model_path:
            past_val_loss = self._restore(maml, opt, scheduler, resume_training=resume)

        for epoch in range(self._epoch, iterations):
            epoch_meta_train_loss = 0.0
            for _ in tqdm(range(iter_per_epoch), dynamic_ncols=True):
                meta_train_losses = []
                opt.zero_grad()
                # One task contains a meta-batch (of size K-Shots + N-Queries) of samples for ONE object class
                for _ in range(batch_size):
                    if self._exit:
                        self._backup()
                        return
                    # Compute the meta-training loss
                    head = maml.clone()
                    # Randomly sample a task (which is created by randomly sampling images, so the
                    # same image sample can appear in several tasks during one epoch, and some
                    # images can not appear during one epoch)
                    meta_batch = self._split_batch(self.dataset.train.sample())
                    inner_loss = self._training_step(
                        meta_batch,
                        head,
                        self.model.features,
                        clip_grad_norm=max_grad_norm,
                        msl=self._msl,
                    )
                    if torch.isnan(inner_loss).any():
                        raise ValueError("Inner loss is Nan!")
                    inner_loss.backward()
                    meta_train_losses.append(inner_loss.detach())

                epoch_meta_train_loss += torch.Tensor(meta_train_losses).mean().item()

                # Average the accumulated gradients and optimize
                for p in maml.parameters():
                    # Some parameters in GraphU-Net are unused but require grad (surely a mistake, but
                    # instead of modifying the original code, this simple check will do).
                    if p.grad is not None:
                        p.grad.data.mul_(1.0 / batch_size)
                # Gradient clipping
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(maml.parameters(), max_grad_norm)
                opt.step()
                if use_scheduler:
                    scheduler.step()

                if self._msl:
                    self._anneal_step_weights()

            epoch_meta_train_loss /= iter_per_epoch

            wandb.log({"meta_train_loss": epoch_meta_train_loss}, step=epoch)
            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {epoch_meta_train_loss:.6f}")

            # ====== Validation ======
            if (epoch + 1) % val_every == 0:
                # Compute the meta-validation loss
                # Go through the entire validation set, which shouldn't be shuffled, and
                # which tasks should not be continuously resampled from!
                meta_val_mse_losses, meta_val_mae_losses = [], []
                for task in tqdm(self.dataset.val, dynamic_ncols=True):
                    head = maml.clone()
                    meta_batch = self._split_batch(task)
                    inner_mse_loss = self._testing_step(
                        meta_batch,
                        head,
                        self.model.features,
                        clip_grad_norm=max_grad_norm,
                    )
                    head = maml.clone()
                    inner_mae_loss = self._testing_step(
                        meta_batch,
                        head,
                        self.model.features,
                        clip_grad_norm=max_grad_norm,
                        compute="mae",
                    )
                    meta_val_mse_losses.append(inner_mse_loss.detach())
                    meta_val_mae_losses.append(inner_mae_loss.detach())
                meta_val_mse_loss = float(
                    torch.Tensor(meta_val_mse_losses).mean().item()
                )
                meta_val_mae_loss = float(
                    torch.Tensor(meta_val_mae_losses).mean().item()
                )

                wandb.log(
                    {
                        "meta_val_mse_loss": meta_val_mse_loss,
                        "meta_val_mae_loss": meta_val_mae_loss,
                    },
                    step=epoch,
                )
                print(f"Meta-validation MSE Loss: {meta_val_mse_loss:.6f}")
                print(f"Meta-validation MAE Loss: {meta_val_mae_loss:.6f}")

                # Model checkpointing
                if meta_val_mse_loss < past_val_loss:
                    state_dicts = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "maml_state_dict": maml.state_dict(),
                        "meta_opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_meta_loss": meta_val_mse_loss,
                    }
                    self._checkpoint(
                        epoch,
                        epoch_meta_train_loss,
                        meta_val_mse_loss,
                        meta_val_mae_loss,
                        state_dicts,
                    )
                    past_val_loss = meta_val_mse_loss
            # ============

            print("============================================")

    def test(
        self,
        batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
    ):
        maml = l2l.algorithms.MAML(
            self.model.head, lr=fast_lr, first_order=self._first_order, allow_unused=True
        )
        all_parameters = list(self.model.features.parameters()) + list(
            maml.parameters()
        )
        opt = torch.optim.Adam(all_parameters, lr=meta_lr, amsgrad=False)
        self._restore(maml, opt, None, resume_training=False)

        meta_mse_losses, meta_mae_losses = [], []
        for task in tqdm(self.dataset.test, dynamic_ncols=True):
            head = maml.clone()
            meta_batch = self._split_batch(task)
            inner_mse_loss = self._testing_step(
                meta_batch,
                head,
                self.model.features,
            )
            head = maml.clone()
            inner_mae_loss = self._testing_step(
                meta_batch,
                head,
                self.model.features,
                compute="mae",
            )
            meta_mse_losses.append(inner_mse_loss.detach())
            meta_mae_losses.append(inner_mae_loss.detach())
        meta_mse_loss = float(
            torch.Tensor(meta_mse_losses).mean().item()
        )
        meta_mae_loss = float(
            torch.Tensor(meta_mae_losses).mean().item()
        )

        print("==========[Test Error]==========")
        print(f"Meta-testing MSE Loss: {meta_mse_loss:.6f}")
        print(f"Meta-testing MAE Loss: {meta_mae_loss:.6f}")
