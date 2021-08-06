#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Model-Agnostic Meta-Learning algorithm.
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.base import BaseTrainer
from collections import namedtuple
from typing import List, Union
from tqdm import tqdm

import matplotlib.pyplot as plt
import learn2learn as l2l
import logging
import torch
import wandb
import os

MetaBatch = namedtuple("MetaBatch", "support query")


class MAMLTrainer(BaseTrainer):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_querries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        assert (
            dataset.k_shots == k_shots
        ), "Dataset's K-shots does not match MAML's K-shots!"
        assert (
            dataset.n_querries == n_querries
        ), "Dataset's N-querries does not match MAML's N-querries!"
        super().__init__(
            model,
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self._k_shots = k_shots
        self._n_querries = n_querries
        self._steps = inner_steps
        self._first_order = first_order

    def _split_batch(self, batch: tuple) -> MetaBatch:
        """
        Separate data batch into adaptation/evalutation sets.
        """
        images, labels_2d, labels_3d = batch
        batch_size = self._k_shots + self._n_querries
        indices = torch.randperm(batch_size)
        support_indices = indices[: self._k_shots]
        query_indices = indices[self._k_shots :]
        return MetaBatch(
            (
                images[support_indices],
                labels_2d[support_indices],
                labels_3d[support_indices],
            ),
            (images[query_indices], labels_2d[query_indices], labels_3d[query_indices]),
        )

    def _restore(self, maml, opt, scheduler, resume_training: bool = True) -> float:
        val_loss = super()._restore(opt, scheduler, resume_training=resume_training)
        checkpoint = torch.load(self._model_path)
        if resume_training and "backup" not in checkpoint.keys():
            maml.load_state_dict(checkpoint["maml_state_dict"])
        return val_loss

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
            self.model, lr=fast_lr, first_order=self._first_order, allow_unused=True
        )
        if optimizer == "adam":
            opt = torch.optim.Adam(maml.parameters(), lr=meta_lr, betas=(0.0, 0.999))
        elif optimizer == "sgd":
            opt = torch.optim.SGD(maml.parameters(), lr=meta_lr)
        else:
            raise ValueError(f"{optimizer} is not a valid outer optimizer")

        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_step_gamma, verbose=True
        )
        scheduler.last_epoch = self._epoch
        past_val_loss = float("+inf")
        if self._model_path:
            past_val_loss = self._restore(maml, opt, scheduler, resume_training=resume)

        for epoch in range(self._epoch, iterations):
            meta_train_losses, meta_val_mse_losses, meta_val_mae_losses = [], [], []
            meta_val_mse_loss, meta_val_mae_loss = 0.0, 0.0
            opt.zero_grad()
            # One task contains a meta-batch (of size K-Shots + N-Queries) of samples for ONE object class
            for _ in tqdm(range(batch_size), dynamic_ncols=True):
                if self._exit:
                    self._backup()
                    return
                # Compute the meta-training loss
                learner = maml.clone()
                meta_batch = self._split_batch(self.dataset.train.sample())
                meta_loss = self._training_step(
                    meta_batch, learner, clip_grad_norm=max_grad_norm
                )
                if torch.isnan(meta_loss).any():
                    raise ValueError("Inner loss is Nan!")
                meta_loss.backward()
                meta_train_losses.append(meta_loss.detach())

                if (epoch + 1) % val_every == 0:
                    # Compute the meta-validation loss
                    learner = maml.clone()
                    meta_batch = self._split_batch(self.dataset.val.sample())
                    inner_mse_loss = self._testing_step(
                        meta_batch, learner, clip_grad_norm=max_grad_norm
                    )
                    learner = maml.clone()
                    inner_mae_loss = self._testing_step(
                        meta_batch, learner, clip_grad_norm=max_grad_norm, compute="mae"
                    )
                    meta_val_mse_losses.append(inner_mse_loss.detach())
                    meta_val_mae_losses.append(inner_mae_loss.detach())
            meta_train_loss = torch.Tensor(meta_train_losses).mean().item()
            if (epoch + 1) % val_every == 0:
                meta_val_mse_loss = float(
                    torch.Tensor(meta_val_mse_losses).mean().item()
                )
                meta_val_mae_loss = float(
                    torch.Tensor(meta_val_mae_losses).mean().item()
                )
            wandb.log({"meta_train_loss": meta_train_loss}, step=epoch)
            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {meta_train_loss:.6f}")
            if (epoch + 1) % val_every == 0:
                wandb.log(
                    {
                        "meta_val_mse_loss": meta_val_mse_loss,
                        "meta_val_mae_loss": meta_val_mae_loss,
                    },
                    step=epoch,
                )
                print(f"Meta-validation MSE Loss: {meta_val_mse_loss:.6f}")
                print(f"Meta-validation MAE Loss: {meta_val_mae_loss:.6f}")
            print("============================================")

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                # Some parameters in GraphU-Net are unused but require grad (surely a mistake, but
                # instead of modifying the original code, this simple check will do).
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / batch_size)
            # Gradient clipping
            if max_grad_norm:
                max_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        maml.parameters(), max_grad_norm
                    ).item()
                )
                print(f"Max gradient norm: {max_norm:.2f}")
            opt.step()
            if use_scheduler:
                scheduler.step()

            # Model checkpointing
            if (epoch + 1) % val_every == 0 and meta_val_mse_loss < past_val_loss:
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
                    meta_train_loss,
                    meta_val_mse_loss,
                    meta_val_mae_loss,
                    state_dicts,
                )
                past_val_loss = meta_val_mse_loss

    def test(
        self,
        batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
    ):
        maml = l2l.algorithms.MAML(
            self.model, lr=fast_lr, first_order=self._first_order, allow_unused=True
        )
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        if self._model_path:
            self._restore(maml, opt, None, resume_training=False)
        meta_test_loss = 0.0
        for task in range(batch_size):
            learner = maml.clone()
            meta_batch = self._split_batch(self.dataset.test.sample())

            print("Support")
            images = meta_batch.support[0]
            for i in range(images.shape[0]):
                image = images[i, :].permute(1, 2, 0).cpu().numpy()
                print(
                    "2D coordinates: ",
                    meta_batch.support[1][i].shape,
                    meta_batch.support[1][i],
                )
                print(
                    "3D coordinates: ",
                    meta_batch.support[2][i].shape,
                    meta_batch.support[2][i],
                )
                plt.imshow(image)
                plt.show()

            print("Query")
            images = meta_batch.query[0]
            for i in range(images.shape[0]):
                image = images[i, :].permute(1, 2, 0).cpu().numpy()
                print(
                    "2D coordinates: ",
                    meta_batch.query[1][i].shape,
                    meta_batch.query[1][i],
                )
                print(
                    "3D coordinates: ",
                    meta_batch.query[2][i].shape,
                    meta_batch.query[2][i],
                )
                plt.imshow(image)
                plt.show()

            meta_loss = self._training_step(meta_batch, learner)
            meta_test_loss += meta_loss.item()
        print("==========[Test Error]==========")
        print(f"Meta-testing Loss: {meta_test_loss:.6f}")
