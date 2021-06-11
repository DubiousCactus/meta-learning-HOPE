#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-Train HOPE-Net
"""

import torchvision.transforms as transforms
import torch.optim as optim
import learn2learn as l2l
import torch.nn as nn
import numpy as np
import torch

from HOPE.utils.options import parse_args_function
from HOPE.utils.model import select_model
from HOPE.utils.dataset import Dataset

from torch.autograd import Variable


class HOPETrainer:
    def __init__(self, dataset_root, lr, lr_step, lr_step_gamma, batch_size, use_cuda=False,
            gpu_number=0, test=False):
        self.model = select_model("hopenet")
        self.use_cuda = use_cuda
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model, device_ids=gpu_number)
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor()])
        self.inner_criterion = nn.MSELoss(reduction='mean')
        # TODO: Add a scheduler in the meta-training loop?
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_step_gamma)
        # self.scheduler.last_epoch = start
        self.lambda_1 = 0.01
        self.lambda_2 = 1
        self.batch_size = batch_size
        self._load_train_val() if not test else self._load_test()

    def _load_train_val(self):
        self.trainset = Dataset(root=self.dataset_root, load_set='train', transform=self.transform)
        self.valset = Dataset(root=self.dataset_root, load_set='val', transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def _load_test(self):
        testset = Dataset(root=self.dataset_root, load_set='test', transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def _training_step(self, batch: tuple, learner, steps: int, shots: int):
        # TODO: Have a batch contain several (input, labels) pairs, and split them in support/query
        # sets
        inputs, labels2d, labels3d = batch
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        # TODO: Do this in the construction of the tasks dataset
        if self.use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            labels3d = labels3d.float().cuda(device=args.gpu_number[0])

        # Adapt the model on the support set
        for step in range(steps):
            # forward + backward + optimize
            outputs2d_init, outputs2d, outputs3d = learner(inputs)
            loss2d_init = self.inner_criterion(outputs2d_init, labels2d)
            loss2d = self.inner_criterion(outputs2d, labels2d)
            loss3d = self.inner_criterion(outputs3d, labels3d)
            support_loss = (self.lambda_1)*loss2d_init + (self.lambda_1)*loss2d + (self.lambda_2)*loss3d
            learner.adapt(support_loss)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, e_outputs2d, e_outputs3d = learner(inputs)
        e_loss2d_init = self.inner_criterion(e_outputs2d_init, labels2d)
        e_loss2d = self.inner_criterion(e_outputs2d, labels2d)
        e_loss3d = self.inner_criterion(e_outputs3d, labels3d)
        query_loss = (self.lambda_1)*e_loss2d_init + (self.lambda_1)*e_loss2d + (self.lambda_2)*e_loss3d

        return query_loss


    def train(self, meta_batch_size: int, iterations: int, fast_lr: float = 0.01,
            meta_lr: float = 0.001, steps: int = 1, shots: int = 10):
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr, first_order=False, allow_unused=True)
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
        batch = next(iter(self.trainloader))
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
                # Some parameters in HOPE-Net are unused but require grad (surely a mistake, but
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


# TODO:
# [ ] Implement the right data loader such that one task = one object (several sequences per object!)
# [ ] Implement MAML learning for the entire HOPENet
# [ ] Implement MAML learning for the feature extractor only (ResNet)
# [ ] Implement MAML learning for Graph U-Net only

def main(args):
    hope_trainer = HOPETrainer(args.input_file, args.learning_rate, args.lr_step,
            args.lr_step_gamma, args.batch_size, use_cuda=args.gpu, gpu_number=args.gpu_number)
    hope_trainer.train(1, 10)



if __name__ == "__main__":
    args = parse_args_function()
    main(args)
