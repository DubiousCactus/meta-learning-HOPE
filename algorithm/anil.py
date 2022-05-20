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

from data.dataset.dex_ycb import DexYCBDatasetTaskLoader
from data.dataset.base import BaseDatasetTaskLoader
from data.custom import CustomDataset
from model.cnn import initialize_weights

from util.utils import compute_curve, plot_curve
from algorithm.maml import MAMLTrainer

from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm

import learn2learn as l2l
import numpy as np
import random
import torch
import wandb

from vendor.bbb import BBBLinear
from vendor.bbb.misc import ModuleWrapper


class BBBEncoder(ModuleWrapper):
    """
    Bayes-By-Backprop encoder for Meta-Regularisation
    """

    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.net = BBBLinear(input_dim, output_dim, bias=True, device=device)


class Head(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hand_only=True):
        super().__init__()
        self._dim = 21 if hand_only else 29
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self._dim * 3),
        )

    def forward(self, x):
        return self.net(x)


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
        beta: float = 1e-7,
        reg_bottleneck_dim: int = 512,
        meta_reg: bool = False,
        task_aug: bool = True,
        hand_only: bool = True,
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
            multi_step_loss=multi_step_loss,
            msl_num_epochs=msl_num_epochs,
            task_aug=task_aug,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self.model: torch.nn.Module = model
        self.head = Head(
            reg_bottleneck_dim if meta_reg else self.model.out_features,
            256,
            hand_only=hand_only,
        )
        self.head.apply(initialize_weights)
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.head = self.head.cuda()
        self.encoder = (
            BBBEncoder(
                self.model.out_features,
                reg_bottleneck_dim,
                device="cuda" if use_cuda else "cpu",
            )
            if meta_reg
            else None
        )
        self._beta = beta
        self._meta_reg = meta_reg
        self._task_aug = task_aug

    def train(
        self,
        batch_size: int = 32,
        iterations: int = 1000,
        fast_lr: float = 0.001,
        meta_lr: float = 0.01,
        optimizer: str = "adam",
        val_every: int = 100,
        resume: bool = True,
        use_scheduler: bool = True,
    ):
        wandb.watch([self.model, self.head])
        maml = l2l.algorithms.MAML(
            self.head,
            lr=fast_lr,
            first_order=self._first_order,
            order_annealing_epoch=self._order_annealing_from_epoch,
            allow_unused=True,
        )
        all_parameters = list(self.model.features.parameters()) + list(
            maml.parameters()
        )
        if optimizer == "adam":
            opt = torch.optim.AdamW(all_parameters, lr=meta_lr, betas=(0.0, 0.999))
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
            T_max=iterations,
            eta_min=0.000001,
            last_epoch=self._epoch - 1,
            verbose=True,
        )
        past_val_loss, meta_val_mse_loss, meta_val_mpjpe = (
            float("+inf"),
            float("+inf"),
            float("+inf"),
        )
        if self._model_path:
            past_val_loss = self._restore(maml, opt, scheduler, resume_training=resume)
            self.head.load_state_dict(torch.load(self._model_path)["head_state_dict"])

        for epoch in range(self._epoch, iterations):
            epoch_meta_train_loss = 0.0
            for _ in tqdm(range(iter_per_epoch), dynamic_ncols=True):
                meta_train_losses = []
                opt.zero_grad()
                # One task contains a meta-batch (of size K-Shots + N-Queries) of samples for ONE object class
                for _ in range(batch_size):
                    if self._exit:
                        state_dicts = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "head_state_dict": self.head.state_dict(),
                            "maml_state_dict": maml.state_dict(),
                            "meta_opt_state_dict": opt.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_meta_loss": meta_val_mse_loss,
                        }
                        self._checkpoint(
                            epoch,
                            epoch_meta_train_loss,
                            meta_val_mse_loss,
                            meta_val_mpjpe,
                            state_dicts,
                        )
                        return
                    # Compute the meta-training loss
                    # Randomly sample a task (which is created by randomly sampling images, so the
                    # same image sample can appear in several tasks during one epoch, and some
                    # images can not appear during one epoch)
                    meta_batch = self._split_batch(self.dataset.train.sample())
                    outer_loss = self._training_step(
                        meta_batch,
                        maml.clone(),
                        self.model.features,
                        epoch,
                        msl=(self._msl and epoch < self._msl_num_epochs),
                    )
                    outer_loss.backward()
                    meta_train_losses.append(outer_loss.detach())

                epoch_meta_train_loss += (
                    torch.Tensor(meta_train_losses).detach().mean().item()
                )

                # Average the accumulated gradients and optimize
                for p in maml.parameters():
                    # Some parameters in GraphU-Net are unused but require grad (surely a mistake, but
                    # instead of modifying the original code, this simple check will do).
                    if p.grad is not None:
                        p.grad.data.mul_(1.0 / batch_size)
                opt.step()

                if self._msl:
                    self._anneal_step_weights()

            if use_scheduler:
                scheduler.step()

            epoch_meta_train_loss /= iter_per_epoch
            del meta_train_losses

            wandb.log({"meta_train_loss": epoch_meta_train_loss}, step=epoch)
            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {epoch_meta_train_loss:.6f}")

            # ====== Validation ======
            if (epoch + 1) % val_every == 0:
                # Compute the meta-validation loss
                # Go through the entire validation set, which shouldn't be shuffled, and
                # which tasks should not be continuously resampled from!
                meta_val_mse_losses, meta_val_mpjpes = [], []
                for task in tqdm(self.dataset.val, dynamic_ncols=True):
                    if self._exit:
                        self._backup()
                        return
                    meta_batch = self._split_batch(task)
                    res = self._testing_step(
                        meta_batch,
                        maml.clone(),
                        self.model.features,
                        epoch,
                        compute=["mse", "pjpe"],
                    )
                    meta_val_mse_losses.append(res["mse"])
                    meta_val_mpjpes.append(res["pjpe"].mean())
                meta_val_mse_loss = float(
                    torch.Tensor(meta_val_mse_losses).mean().item()
                )
                meta_val_mpjpe = float(torch.Tensor(meta_val_mpjpes).mean().item())

                wandb.log(
                    {
                        "meta_val_mse_loss": meta_val_mse_loss,
                        "meta_val_mpjpe": meta_val_mpjpe,
                    },
                    step=epoch,
                )
                print(f"Meta-validation MSE Loss: {meta_val_mse_loss:.6f}")
                print(f"Meta-validation MPJPE: {meta_val_mpjpe:.6f}")

                # Model checkpointing
                if meta_val_mse_loss < past_val_loss:
                    state_dicts = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "head_state_dict": self.head.state_dict(),
                        "maml_state_dict": maml.state_dict(),
                        "meta_opt_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_meta_loss": meta_val_mse_loss,
                    }
                    self._checkpoint(
                        epoch,
                        epoch_meta_train_loss,
                        meta_val_mse_loss,
                        meta_val_mpjpe,
                        state_dicts,
                    )
                    past_val_loss = meta_val_mse_loss
            # ============

            print("============================================")

    def test(
        self,
        batch_size: int = 16,  # Unused
        runs: int = 1,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        visualize: bool = False,
        plot: bool = False,
    ):
        maml = l2l.algorithms.MAML(
            self.head,
            lr=fast_lr,
            first_order=self._first_order,
            allow_unused=True,
        )
        all_parameters = list(self.model.features.parameters()) + list(
            maml.parameters()
        )
        # The optimiser is needed for the restore function because I'm too lazy to make it optional
        opt = torch.optim.Adam(all_parameters, lr=meta_lr, amsgrad=False)
        if self._model_path:
            self._restore(maml, opt, None, resume_training=False)
            self.head.load_state_dict(torch.load(self._model_path)["head_state_dict"])

        avg_mpjpe, avg_mpcpe, avg_auc_pck, avg_auc_pcp = 0.0, 0.0, 0.0, 0.0
        thresholds = torch.linspace(10, 100, (100 - 10) // 5 + 1)
        min_mpjpe = float("+inf")
        compute = ["pjpe"]
        if not self._hand_only:
            compute.append("pcpe")

        for i in range(runs):
            print(f"===============[Run {i+1}/{runs}]==============")
            MPJPEs, MPCPEs = [], []
            PJPEs, PCPEs = [], []

            for task in tqdm(self.dataset.test, dynamic_ncols=True):
                if self._exit:
                    return
                meta_batch = self._split_batch(task)
                res = self._testing_step(
                    meta_batch,
                    maml.clone(),
                    self.model.features,
                    compute=compute,
                )
                PJPEs.append(res["pjpe"])
                if visualize:
                    self._testing_step_vis(
                        meta_batch, maml.clone(), self.model.features
                    )
                MPJPEs.append(res["pjpe"].mean())
                if not self._hand_only:
                    PCPEs.append(res["pcpe"])
                    MPCPEs.append(res["pcpe"].mean())

            print("-> Computing PCK curves...")
            # Compute the PCK curves (hand joints)
            auc, pck = compute_curve(PJPEs, thresholds, 21)
            avg_auc_pck += auc
            mpjpe = float(torch.Tensor(MPJPEs).mean().item())
            avg_mpjpe += mpjpe
            if not self._hand_only:
                # Compute the PCK curves (hand joints)
                auc, pcp = compute_curve(PCPEs, thresholds, 8)
                avg_auc_pcp += auc
                mpcpe = float(torch.Tensor(MPCPEs).mean().item())
                avg_mpcpe += mpcpe

            if mpjpe < min_mpjpe and plot:
                plot_curve(pck, thresholds, "anil_pck.png")
                min_mpjpe = mpjpe
                if not self._hand_only:
                    plot_curve(pcp, thresholds, "anil_pcp.png")
            print(f"=======================================")
        avg_mpjpe /= float(runs)
        avg_auc_pck /= float(runs)
        print(f"\n\n==========[Test Error (avg of {runs})]==========")
        print(f"Mean Per Joint Pose Error: {avg_mpjpe:.6f}")
        print(f"Mean Area Under Curve for PCK: {avg_auc_pck:.6f}")
        if not self._hand_only:
            avg_mpcpe /= float(runs)
            avg_auc_pcp /= float(runs)
            print(f"Mean Per Corner Pose Error: {avg_mpcpe:.6f}")
            print(f"Mean Area Under Curve for PCP: {avg_auc_pcp:.6f}")

    def analyse_inner_gradients(
        self,
        data_loader: DexYCBDatasetTaskLoader,
        fast_lr: float = 0.01,
        n_tasks: int = 10,
    ):
        """
        Compute the average inner gradient norms of N tasks for K random objects.
        Used for Table 2 in the Analysis section.
        """
        maml = l2l.algorithms.MAML(
            self.head,
            lr=fast_lr,
            first_order=self._first_order,
            allow_unused=True,
        )
        all_parameters = list(self.model.features.parameters()) + list(
            maml.parameters()
        )
        # The optimiser is needed for the restore function because I'm too lazy to make it optional
        opt = torch.optim.Adam(all_parameters)
        if self._model_path:
            self._restore(maml, opt, None, resume_training=False)
            self.head.load_state_dict(torch.load(self._model_path)["head_state_dict"])

        samples = data_loader.make_raw_dataset()
        # Only keep the test set that this model was trained for (so we don't have train/test
        # overlap)
        keys = list(samples.copy().keys())
        for category_id in keys:
            if category_id not in data_loader.split_categories["test"]:
                # I'm aware of this stupidity... See data/dataset/dex_ycb.py, somewhere in the
                # make_dataset() function for explanations.
                del samples[keys[category_id]]

        # Reproducibility:
        np.random.seed(1995)
        torch.manual_seed(1995)
        random.seed(1995)
        # For pytorch's multi-process data loading:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(1995)

        random_obj = list(np.random.choice(list(samples.keys()), size=5, replace=False))
        print(
            f"[*] Analysing gradients for {', '.join([data_loader.obj_labels[i] for i in random_obj])}"
        )
        obj_norms = {}
        for obj_id in random_obj:
            if self._exit:
                return
            # Still using the custom dataset because of the preprocessing (root alignment)
            # Set object_as_task=False because we pass it a list and not a dict
            task = CustomDataset(
                samples[obj_id],
                img_transform=BaseDatasetTaskLoader._img_transform,
                object_as_task=False,
                hand_only=self._hand_only,
            )
            # I'm not using learn2learn because there's only one class so it wouldn't work
            dataset = DataLoader(
                task,
                batch_size=self._k_shots + self._n_queries,
                shuffle=True,
                num_workers=0,  # Disable multiple workers to avoid issues of randomness, and because it'll be faster since we're re-creating dataloaders
                worker_init_fn=seed_worker,  # But better be cautious haha
                generator=g,
            )
            step_norms = {i: 0.0 for i in range(self._steps)}
            for i, task in tqdm(enumerate(dataset), dynamic_ncols=True, total=n_tasks):
                if i == n_tasks:
                    break
                assert len(task[0]) == (
                    self._n_queries + self._k_shots
                ), "Batch not full. Try reducing the number of tasks to analyse!"
                meta_batch = self._split_batch(task)

                s_inputs, _, s_labels3d = meta_batch.support
                if self._use_cuda:
                    s_inputs = s_inputs.float().cuda(device=self._gpu_number)
                    s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)

                with torch.no_grad():
                    s_inputs = self.model.features(s_inputs)
                head = maml.clone()
                # Adapt the model on the support set
                for step in range(self._steps):
                    # forward + backward + optimize
                    joints = head(s_inputs).view(-1, self._dim, 3)
                    joints -= (
                        joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1)
                    )  # Root alignment
                    support_loss = self.inner_criterion(joints, s_labels3d)
                    grad_norm = head.adapt(
                        support_loss, epoch=None, return_grad_norm=True
                    )
                    step_norms[step] += float(grad_norm.detach().cpu().numpy())
            for step, norm in step_norms.items():
                step_norms[step] = norm / n_tasks
            obj_norms[obj_id] = step_norms
        for obj_id, step_norms in obj_norms.items():
            print(f"Object {data_loader.obj_labels[obj_id][4:]}:")
            for step, norm in step_norms.items():
                print(f"\tStep {step}: {norm:.2f}")
            print()
