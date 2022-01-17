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
from sklearn.manifold import TSNE
from typing import List
from tqdm import tqdm

import matplotlib.pyplot as plt
import learn2learn as l2l
import seaborn as sns
import pandas as pd
import numpy as np
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
                meta_val_mse_losses, meta_val_mae_losses = [], []
                for task in tqdm(self.dataset.val, dynamic_ncols=True):
                    if self._exit:
                        self._backup()
                        return
                    meta_batch = self._split_batch(task)
                    inner_mse_loss = self._testing_step(
                        meta_batch,
                        maml.clone(),
                        self.model.features,
                        epoch,
                    )
                    inner_mae_loss = self._testing_step(
                        meta_batch,
                        maml.clone(),
                        self.model.features,
                        epoch,
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
                del meta_val_mae_losses
                del meta_val_mse_losses

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
        opt = torch.optim.Adam(all_parameters, lr=meta_lr, amsgrad=False)
        if self._model_path:
            self._restore(maml, opt, None, resume_training=False)

        params, y = [], []
        for task in tqdm(self.dataset.test, dynamic_ncols=True):
            if self._exit:
                return
            head = maml.clone()
            features = self.model.features
            meta_batch = self._split_batch(task)
            s_inputs, obj_label, s_labels3d = meta_batch.support
            q_inputs, obj_label, q_labels3d = meta_batch.query
            if self._use_cuda:
                s_inputs = s_inputs.float().cuda(device=self._gpu_number)
                s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
                q_inputs = q_inputs.float().cuda(device=self._gpu_number)
                q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

            with torch.no_grad():
                s_inputs = features(s_inputs)
            # Adapt the model on the support set
            for _ in range(self._steps):
                # forward + backward + optimize
                joints = head(s_inputs).view(-1, 29, 3)
                joints -= (
                    joints[:, 0, :].unsqueeze(dim=1).expand(-1, 29, -1)
                )  # Root alignment
                support_loss = self.inner_criterion(joints, s_labels3d)
                head.adapt(support_loss, epoch=None)

            # for name, p in head.named_parameters():
            #     if "module.0.weight" in name:
            #         params.append(p.detach().flatten().cpu().numpy())
            #         y.append(1)
            net_params = []
            for p in head.parameters():
                net_params.append(p.detach().flatten())#.cpu().numpy())
            params.append(torch.cat(net_params).cpu().numpy())
            y.append(int(obj_label[0]))

        print("[*] Running t-SNE...")
        params = np.array(params)
        # params = torch.cat(params).cpu().numpy()
        print(params.shape)
        embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(params)
        tsne_result_df = pd.DataFrame({'tsne_1': embedded[:,0], 'tsne_2': embedded[:,1], 'label': y})
        tsne_result_df.to_pickle(f"tsne_dataframe_{self.dataset.held_out}.pkl")
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120,
                palette="deep")
        lim = (embedded.min()-5, embedded.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.savefig(f"t-sne_{self.dataset.held_out}.png")
        plt.show()
