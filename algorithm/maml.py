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

from algorithm.base import BaseTrainer
from data.utils import DatasetFactory

import learn2learn as l2l
import torch


class MAMLTrainer(BaseTrainer):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_root: str,
        batch_size: int,
        k_shots: int,
        use_cuda: int = False,
        gpu_number: int = 0,
        test_mode: bool = False,
        object_as_task: bool = True,
    ):
        dataset = DatasetFactory.make_data_loader(
            dataset_name, dataset_root, batch_size, test_mode, False, k_shots
        )
        super().__init__(
            model_name,
            dataset,
            batch_size,
            use_cuda=use_cuda,
            gpu_number=gpu_number,
            test_mode=test_mode,
        )

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        raise NotImplementedError("_training_step() not implemented!")

    def train(
        self,
        meta_batch_size: int = 16,
        iterations: int = 1000,
        fast_lr: float = 0.0001,
        meta_lr: float = 0.0001,
        steps: int = 1,
        shots: int = 10,
    ):
        maml = l2l.algorithms.MAML(
            self.model, lr=fast_lr, first_order=False, allow_unused=True
        )
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        # TODO: Test by plotting all images in the batch!
        batch = self.dataset.train.sample()
        print("Batch:", batch)
        batch = self.dataset.train.sample()
        print("Batch:", batch)
        for iteration in range(iterations):
            opt.zero_grad()
            meta_train_loss = 0.0
            meta_val_loss = 0.0
            for task in range(meta_batch_size):
                # Compute the meta-training loss
                learner = maml.clone()
                # batch = tasks_set.train.sample()
                inner_loss = self._training_step(batch, learner, steps, shots)
                inner_loss.backward()
                meta_train_loss += inner_loss.item()

                # Compute the meta-validation loss
                # leaner = maml.clone()
                # batch = tasks_set.validation.sample()
                # inner_loss = self._training_step(batch, learner, steps, shots)
                # meta_val_loss += inner_loss.item()
            print(f"==========[Iteration {iteration}]==========")
            print(f"Meta-training Loss: {meta_train_loss/meta_batch_size:.6f}")
            print(f"Meta-validation Loss: {meta_val_loss/meta_batch_size:.6f}")
            print("============================================")

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                # Some parameters in GraphU-Net are unused but require grad (surely a mistake, but
                # instead of modifying the original code, this simple check will do).
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

    def test(
        self,
        meta_batch_size: int = 16,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        steps: int = 1,
        shots: int = 10,
    ):
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr, first_order=True)
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        meta_test_loss = 0.0
        for task in range(meta_batch_size):
            learner = maml.clone()
            batch = tasks_set.test.sample()
            inner_loss = self._training_step(batch, learner, steps, shots)
            meta_test_loss += inner_loss.item()
        print("==========[Test Error]==========")
        print(f"Meta-testing Loss: {meta_test_loss:.6f}")
