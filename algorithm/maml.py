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
from util.utils import compute_curve, plot_curve
from algorithm.base import BaseTrainer
from collections import namedtuple
from typing import List, Optional, Union
from tqdm import tqdm

import learn2learn as l2l
import torch
import wandb

MetaBatch = namedtuple("MetaBatch", "support query")


class MAMLTrainer(BaseTrainer):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries: int,
        inner_steps: int,
        model_path: str = None,
        first_order: bool = False,
        multi_step_loss: bool = True,
        msl_num_epochs: int = 1000,
        task_aug: Optional[str] = None,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        assert (
            dataset.k_shots == k_shots
        ), "Dataset's K-shots does not match MAML's K-shots!"
        assert (
            dataset.n_queries == n_queries
        ), "Dataset's N-queries does not match MAML's N-queries!"
        super().__init__(
            model,
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self._k_shots = k_shots
        self._n_queries = n_queries
        self._steps = inner_steps
        self._first_order = first_order
        self._step_weights = torch.ones(inner_steps) * (1.0 / inner_steps)
        self._msl = multi_step_loss
        self._msl_num_epochs = msl_num_epochs
        self._msl_decay_rate = 1.0 / self._steps / msl_num_epochs
        self._msl_min_value_for_non_final_losses = torch.tensor(0.03 / self._steps)
        self._msl_max_value_for_final_loss = 1.0 - (
            (self._steps - 1) * self._msl_min_value_for_non_final_losses
        )
        self._order_annealing_from_epoch = 50
        self._task_aug = task_aug
        self._task_aug_noise_values = 4  # TODO: Add to config for hyperparam search

    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self._steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def _split_batch(self, batch: tuple) -> MetaBatch:
        """
        Separate data batch into adaptation/evalutation sets.
        """
        images, labels_2d, labels_3d = batch
        batch_size = self._k_shots + self._n_queries
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
        optimizer: str = "adam",
        val_every: int = 100,
        resume: bool = True,
        use_scheduler: bool = True,
    ):
        wandb.watch(self.model)
        maml = l2l.algorithms.MAML(
            self.model,
            lr=fast_lr,
            first_order=self._first_order,
            allow_unused=True,
            order_annealing_epoch=self._order_annealing_from_epoch,
        )
        if optimizer == "adam":
            opt = torch.optim.AdamW(maml.parameters(), lr=meta_lr, betas=(0.0, 0.999))
        elif optimizer == "sgd":
            opt = torch.optim.SGD(maml.parameters(), lr=meta_lr)
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
        past_val_loss = float("+inf")
        if self._model_path:
            past_val_loss = self._restore(maml, opt, scheduler, resume_training=resume)

        for epoch in range(self._epoch, iterations):
            epoch_meta_train_loss = 0.0
            for _ in tqdm(range(iter_per_epoch), dynamic_ncols=True):
                meta_train_losses = []
                opt.zero_grad()
                maml.zero_grad()
                # One task contains a meta-batch (of size K-Shots + N-Queries) of samples for ONE object class
                for _ in range(batch_size):
                    if self._exit:
                        self._backup()
                        return
                    # Compute the meta-training loss
                    meta_batch = self._split_batch(self.dataset.train.sample())
                    meta_loss = self._training_step(
                        meta_batch,
                        maml.clone(),
                        epoch,
                        msl=(self._msl and epoch < self._msl_num_epochs),
                    )
                    meta_loss.backward()
                    meta_train_losses.append(meta_loss.detach())

                epoch_meta_train_loss += torch.Tensor(meta_train_losses).mean().item()

                # Average the accumulated gradients and optimize
                with torch.no_grad():
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
            wandb.log({"meta_train_loss": epoch_meta_train_loss}, step=epoch)

            print(f"==========[Epoch {epoch}]==========")
            print(f"Meta-training Loss: {epoch_meta_train_loss:.6f}")

            # ======= Validation ========
            if (epoch + 1) % val_every == 0:
                # Compute the meta-validation loss
                # Go through the entire validation set, which shouldn't be shuffled, and
                # which tasks should not be continuously resampled from!
                meta_val_mse_losses, meta_val_mae_losses = [], []
                for task in tqdm(self.dataset.val, dynamic_ncols=True):
                    meta_batch = self._split_batch(task)
                    inner_mse_loss = self._testing_step(meta_batch, maml.clone(), epoch)
                    inner_mae_loss = self._testing_step(
                        meta_batch, maml.clone(), epoch, compute="mae"
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

                print("============================================")

    def test(
        self,
        batch_size: int = 16,
        runs: int = 1,
        fast_lr: float = 0.01,
        meta_lr: float = 0.001,
        visualize: bool = False,
        plot: bool = False,
    ):
        maml = l2l.algorithms.MAML(
            self.model, lr=fast_lr, first_order=self._first_order, allow_unused=True
        )
        opt = torch.optim.Adam(maml.parameters(), lr=meta_lr, amsgrad=False)
        if self._model_path:
            self._restore(maml, opt, None, resume_training=False)

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
                    compute=compute,
                )
                PJPEs.append(res["pjpe"])
                if visualize:
                    self._testing_step_vis(meta_batch, maml.clone())
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
