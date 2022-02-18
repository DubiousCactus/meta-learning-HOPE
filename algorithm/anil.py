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
from util.utils import compute_curve, plot_curve
from algorithm.maml import MAMLTrainer
from typing import List
from tqdm import tqdm

import learn2learn as l2l
import torch
import wandb


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
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self.model: torch.nn.Module = model
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

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
            self.model.head,
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
        past_val_loss, meta_val_mse_loss, meta_val_mpjpe = float("+inf"), float("+inf"), float("+inf")
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
                            meta_val_mpjpe,
                            state_dicts,
                        )
                        return
                    # Compute the meta-training loss
                    # Randomly sample a task (which is created by randomly sampling images, so the
                    # same image sample can appear in several tasks during one epoch, and some
                    # images can not appear during one epoch)
                    meta_batch = self._split_batch(self.dataset.train.sample())
                    inner_loss = self._training_step(
                        meta_batch,
                        maml.clone(),
                        self.model.features,
                        epoch,
                        msl=(self._msl and epoch < self._msl_num_epochs),
                    )
                    inner_loss.backward()
                    meta_train_losses.append(inner_loss.detach())

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
                meta_val_mpjpe = float(
                    torch.Tensor(meta_val_mpjpes).mean().item()
                )

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
            self.model.head,
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

        avg_mpjpe, avg_mpcpe, avg_auc_pck, avg_auc_pcp = 0.0, 0.0, 0.0, 0.0
        thresholds = torch.linspace(10, 100, (100-10)//5+1)
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
