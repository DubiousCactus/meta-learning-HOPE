#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-Training.
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.base import BaseTrainer
from collections import namedtuple

import matplotlib.pyplot as plt
import learn2learn as l2l
import torch
import os

MetaBatch = namedtuple("MetaBatch", "support query")


class MAMLTrainer(BaseTrainer):
    def __init__(
        self,
        model_name: str,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_querries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
        object_as_task: bool = True,
    ):
        assert (
            dataset.k_shots == k_shots
        ), "Dataset's K-shots does not match MAML's K-shots!"
        assert (
            dataset.n_querries == n_querries
        ), "Dataset's N-querries does not match MAML's N-querries!"
        super().__init__(
            model_name,
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )
        self._k_shots = k_shots
        self._n_querries = n_querries
        self._steps = inner_steps
        self._first_order = first_order

    def _training_step(self, support: tuple, query: tuple, learner):
        raise NotImplementedError("_training_step() not implemented!")

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

    def _restore(self, maml, opt, scheduler, resume_training: bool = True):
        checkpoint = torch.load(self._model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        maml.load_state_dict(checkpoint["maml_state_dict"])
        opt.load_state_dict(checkpoint["meta_opt_state_dict"])
        if resume_training:
            self._epoch = checkpoint["epoch"] + 1
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            return checkpoint["val_meta_loss"]

    def train(
        self,
        meta_batch_size: int = 32,
        iterations: int = 1000,
        fast_lr: float = 0.001,
        meta_lr: float = 0.01,
        lr_step: int = 100,
        lr_step_gamma: float = 0.5,
        save_every: int = 100,
        val_every: int = 100,
        resume: bool = True,
    ):
        maml = l2l.algorithms.MAML(
            self.model, lr=fast_lr, first_order=self._first_order, allow_unused=True
        )
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_step_gamma, verbose=True
        )
        scheduler.last_epoch = self._epoch
        past_val_loss = float("+inf")
        if self._model_path:
            saved_val_loss = self._restore(maml, opt, scheduler, resume_training=resume)
            if resume_training:
                past_val_loss = saved_val_loss
        for iteration in range(self._epoch, iterations):
            opt.zero_grad()
            meta_train_loss = 0.0
            meta_val_loss = 0.0
            # One task contains a meta-batch (of size K-Shots + N-Queries) of samples for ONE object class
            for task in range(meta_batch_size):
                # Compute the meta-training loss
                learner = maml.clone()
                meta_batch = self._split_batch(self.dataset.train.sample())
                inner_loss = self._training_step(meta_batch, learner)
                inner_loss.backward()
                meta_train_loss += inner_loss.item()

                if iteration % val_every == 0:
                    # Compute the meta-validation loss
                    leaner = maml.clone()
                    meta_batch = self._split_batch(self.dataset.val.sample())
                    inner_loss = self._training_step(meta_batch, learner)
                    meta_val_loss += inner_loss.item()
            meta_train_loss = meta_train_loss / meta_batch_size
            if iteration % val_every == 0:
                meta_val_loss = meta_val_loss / meta_batch_size
            print(f"==========[Iteration {iteration}]==========")
            print(f"Meta-training Loss: {meta_train_loss:.6f}")
            if iteration % val_every == 0:
                print(f"Meta-validation Loss: {meta_val_loss:.6f}")
            print("============================================")

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                # Some parameters in GraphU-Net are unused but require grad (surely a mistake, but
                # instead of modifying the original code, this simple check will do).
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
            scheduler.step()

            # Model checkpointing
            if iteration % save_every == 0 and meta_val_loss < past_val_loss:
                print(f"-> Saving model to {self._checkpoint_path}...")
                torch.save(
                    {
                        "epoch": iteration,
                        "model_state_dict": self.model.state_dict(),
                        "maml_state_dict": maml.state_dict(),
                        "meta_opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_meta_loss": meta_val_loss,
                    },
                    os.path.join(
                        self._checkpoint_path,
                        f"epoch_{iteration}_train_loss-{meta_train_loss:.6f}_val_loss-{meta_val_loss:.6f}.tar",
                    ),
                )
                past_val_loss = meta_val_loss

    def test(
        self,
        meta_batch_size: int = 16,
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
        for task in range(meta_batch_size):
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

            inner_loss = self._training_step(meta_batch, learner)
            meta_test_loss += inner_loss.item()
        print("==========[Test Error]==========")
        print(f"Meta-testing Loss: {meta_test_loss:.6f}")
