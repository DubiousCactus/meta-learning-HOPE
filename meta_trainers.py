#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-Training wrappers.
"""

import torch.optim as optim
import learn2learn as l2l
import torch.nn as nn
import torch

from HOPE.utils.model import select_model

from dataset import DatasetFactory, BaseDatasetTaskLoader

from torch.autograd import Variable
from abc import ABC


class BaseTrainer(ABC):
    def __init__(self, model_name: str, dataset: BaseDatasetTaskLoader, lr: float,
            lr_step: float, lr_step_gamma: float, batch_size: int, use_cuda: int = False,
            gpu_number: int = 0, test_mode: bool = False):
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number
        self.model = select_model(model_name)
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids=gpu_number)
        self.dataset = dataset
        self.inner_criterion = nn.MSELoss()
        # TODO: Add a scheduler in the meta-training loop?
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step,
        # gamma=lr_step_gamma)
        # self.scheduler.last_epoch = start
        self._lambda1 = 0.01
        self._lambda2 = 1
        self.batch_size = batch_size

    def _load_train_val(self, dataset_root: str, batch_size: int):
        trainset = Dataset(root=dataset_root, load_set='train', transform=self._transform)
        valset = Dataset(root=dataset_root, load_set='val', transform=self._transform)
        train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                shuffle=True, num_workers=16)
        val_data_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                shuffle=False, num_workers=8)
        return train_data_loader, val_data_loader

    def _load_test(self, dataset_root: str, batch_size: int):
        testset = Dataset(root=dataset_root, load_set='test', transform=self._transform)
        test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                shuffle=False, num_workers=8)
        return test_data_loader


class MAMLTrainer(BaseTrainer):
    def __init__(self, model_name: str, dataset_name: str, dataset_root: str, lr: float,
            lr_step: float, lr_step_gamma: float, batch_size: int, use_cuda: int = False,
            gpu_number: int = 0, test_mode: bool = False, object_as_task: bool = True):
        dataset = DatasetFactory.make_data_loader(dataset_name, dataset_root, batch_size,
                test_mode, object_as_task)
        super().__init__(model_name, dataset, lr, lr_step, lr_step_gamma, batch_size,
                use_cuda=use_cuda, gpu_number=gpu_number, test_mode=test_mode)

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        raise NotImplementedError("_training_step() not implemented!")

    def train(self, meta_batch_size: int, iterations: int, fast_lr: float = 0.01,
            meta_lr: float = 0.001, steps: int = 1, shots: int = 10):
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr, first_order=False, allow_unused=True)
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        batch = next(iter(self.dataset.train))
        for iteration in range(iterations):
            opt.zero_grad()
            meta_train_loss = .0
            meta_val_loss = .0
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

    def test(self, meta_batch_size: int, fast_lr: float = 0.1, meta_lr: float = 0.001,
            steps: int = 5, shots: int = 10):
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr, first_order=True)
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        meta_test_loss = .0
        for task in range(meta_batch_size):
            learner = maml.clone()
            batch = tasks_set.test.sample()
            inner_loss = self._training_step(batch, learner, steps, shots)
            meta_test_loss += inner_loss.item()
        print("==========[Test Error]==========")
        print(f"Meta-testing Loss: {meta_test_loss:.6f}")


class HOPETrainer(MAMLTrainer):
    def __init__(self, dataset_name: str, dataset_root: str, lr: float,
            lr_step: float, lr_step_gamma: float, batch_size: int, use_cuda: int = False,
            gpu_number: int = 0, test_mode: bool = False):
        super().__init__("hopenet", dataset_name, dataset_root, lr, lr_step, lr_step_gamma,
                batch_size, use_cuda=use_cuda, gpu_number=gpu_number, test_mode=test_mode)

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        inputs, labels2d, labels3d = batch
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        # TODO: Do this in the construction of the tasks dataset
        if self._use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=self._gpu_number[0])
            labels2d = labels2d.float().cuda(device=self._gpu_number[0])
            labels3d = labels3d.float().cuda(device=self._gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs2d_init, outputs2d, outputs3d = learner(inputs)
            loss2d_init = self.inner_criterion(outputs2d_init, labels2d)
            loss2d = self.inner_criterion(outputs2d, labels2d)
            loss3d = self.inner_criterion(outputs3d, labels3d)
            support_loss = ((self._lambda1)*loss2d_init + (self._lambda1)*loss2d +
                    (self._lambda2)*loss3d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, e_outputs2d, e_outputs3d = learner(inputs)
        e_loss2d_init = self.inner_criterion(e_outputs2d_init, labels2d)
        e_loss2d = self.inner_criterion(e_outputs2d, labels2d)
        e_loss3d = self.inner_criterion(e_outputs3d, labels3d)
        query_loss = ((self._lambda1)*e_loss2d_init + (self._lambda1)*e_loss2d +
                (self._lambda2)*e_loss3d)
        return query_loss


class ResnetTrainer(MAMLTrainer):
    def __init__(self, dataset_name: str, dataset_root: str, lr: float,
            lr_step: float, lr_step_gamma: float, batch_size: int, use_cuda: int = False,
            gpu_number: int = 0, test_mode: bool = False):
        super().__init__("resnet10", dataset_name, dataset_root, lr, lr_step, lr_step_gamma,
                batch_size, use_cuda=use_cuda, gpu_number=gpu_number, test_mode=test_mode)

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        inputs, labels2d, _ = batch
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)

        # TODO: Do this in the construction of the tasks dataset
        if self._use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=self._gpu_number[0])
            labels2d = labels2d.float().cuda(device=self._gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs2d_init, _ = learner(inputs)
            support_loss = self.inner_criterion(outputs2d_init, labels2d)
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, _ = learner(inputs)
        query_loss = self.inner_criterion(e_outputs2d_init, labels2d)
        return query_loss


