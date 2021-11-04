#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Part-specific training wrappers.
"""

from data.dataset.base import BaseDatasetTaskLoader
from algorithm.maml import MAMLTrainer, MetaBatch
from algorithm.regular import RegularTrainer
from algorithm.anil import ANILTrainer

from HOPE.models.graphunet import GraphNet
from HOPE.utils.model import select_model

from model.graphnet import GraphUNetBatchNorm, GraphNetBatchNorm
from model.hopenet import HOPENet, GraphNetwResNet
from model.cnn import ResNet, MobileNet

from util.utils import load_state_dict

from typing import List
from tqdm import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import torch


class MAML_HOPETrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries,
        inner_steps: int,
        cnn_def: str,
        resnet_path: str,
        graphnet_path: str,
        graphunet_path: str,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
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

    def _training_step(
        self, batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, s_labels2d, s_labels3d = batch.support
        q_inputs, q_labels2d, q_labels3d = batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs2d_init, outputs2d, outputs3d = learner(s_inputs)
            nan1 = torch.isnan(outputs2d_init).any()
            nan2 = torch.isnan(outputs2d).any()
            nan3 = torch.isnan(outputs3d).any()
            if nan1 or nan2 or nan3:
                print(f"Support outputs contains NaN!")
            loss2d_init = self.inner_criterion(outputs2d_init, s_labels2d)
            loss2d = self.inner_criterion(outputs2d, s_labels2d)
            loss3d = self.inner_criterion(outputs3d, s_labels3d)
            support_loss = (
                (self._lambda1) * loss2d_init
                + (self._lambda1) * loss2d
                + (self._lambda2) * loss3d
            )
            learner.adapt(support_loss, clip_grad_max_norm=clip_grad_norm)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, e_outputs2d, e_outputs3d = learner(q_inputs)
        nan1 = torch.isnan(e_outputs2d_init).any()
        nan2 = torch.isnan(e_outputs2d).any()
        nan3 = torch.isnan(e_outputs3d).any()
        if nan1 or nan2 or nan3:
            print(f"Query outputs contains NaN!")
        e_loss2d_init = criterion(e_outputs2d_init, q_labels2d)
        e_loss2d = criterion(e_outputs2d, q_labels2d)
        e_loss3d = criterion(e_outputs3d, q_labels3d)
        query_loss = (
            (self._lambda1) * e_loss2d_init
            + (self._lambda1) * e_loss2d
            + (self._lambda2) * e_loss3d
        )
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        return self._training_step(meta_batch, learner, clip_grad_norm, compute)


class MAML_CNNTrainer(MAMLTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries,
        inner_steps: int,
        cnn_def: str,
        model_path: str = None,
        first_order: bool = False,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="18", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="34", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        super().__init__(
            cnn,
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

    def _training_step(
        self, batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, s_labels2d, _ = batch.support
        q_inputs, q_labels2d, _ = batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs2d_init, _ = learner(s_inputs)
            if torch.isnan(outputs2d_init).any():
                print(f"Support outputs contains NaN!")
            support_loss = self.inner_criterion(outputs2d_init, s_labels2d)
            learner.adapt(support_loss, clip_grad_max_norm=clip_grad_norm)

        # Evaluate the adapted model on the query set
        e_outputs2d_init, _ = learner(q_inputs)
        if torch.isnan(e_outputs2d_init).any():
            print(f"Query outputs contains NaN!")
        query_loss = criterion(e_outputs2d_init, q_labels2d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        return self._training_step(meta_batch, learner, clip_grad_norm, compute)


class ANIL_CNNTrainer(ANILTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        k_shots: int,
        n_queries,
        inner_steps: int,
        cnn_def: str,
        model_path: str = None,
        first_order: bool = False,
        multi_step_loss: bool = True,
        msl_num_epochs: int = 1000,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="18", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="34", pretrained=True)
        elif cnn_def == "resnet50":
            cnn = ResNet(model="50", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        super().__init__(
            cnn,
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

    def _training_step(
        self,
        batch: MetaBatch,
        head,
        features,
        epoch,
        clip_grad_norm=None,
        compute="mse",
        msl=True,
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, _, s_labels3d = batch.support
        q_inputs, _, q_labels3d = batch.query
        query_loss = .0
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        s_inputs = features(s_inputs)
        q_inputs_features = features(q_inputs)
        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            outputs3d = head(s_inputs).view(-1, 29, 3)
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            head.adapt(support_loss, epoch=epoch, clip_grad_max_norm=clip_grad_norm)
            if msl:  # Multi-step loss
                q_outputs3d = head(q_inputs_features).view(-1, 29, 3)
                query_loss += self._step_weights[step] * criterion(
                    q_outputs3d, q_labels3d
                )

        # Evaluate the adapted model on the query set
        if not msl:
            q_outputs3d = head(q_inputs_features).view(-1, 29, 3)
            query_loss = criterion(q_outputs3d, q_labels3d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, head, features, epoch=None, clip_grad_norm=None, compute="mse"
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        s_inputs, _, s_labels3d = meta_batch.support
        q_inputs, _, q_labels3d = meta_batch.query
        if self._use_cuda:
            s_inputs = s_inputs.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_inputs = q_inputs.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        s_inputs = features(s_inputs)
        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs3d = head(s_inputs).view(-1, 29, 3)
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            head.adapt(support_loss, epoch=epoch, clip_grad_max_norm=clip_grad_norm)

        with torch.no_grad():
            q_inputs = features(q_inputs)
            q_outputs3d = head(q_inputs).view(-1, 29, 3)
        return criterion(q_outputs3d, q_labels3d)


class MAML_GraphUNetTrainer(MAMLTrainer):
    def __init__(
        self,
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
            "graphunet",
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

    def _training_step(
        self, batch: MetaBatch, learner, clip_grad_norm=None, compute="mse", msl=True
    ):
        criterion = self.inner_criterion
        if compute == "mae":
            criterion = F.l1_loss
        _, s_labels2d, s_labels3d = batch.support
        _, q_labels2d, q_labels3d = batch.query
        query_loss = .0
        if self._use_cuda:
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # with torch.no_grad():
        # avg_norm = []
        # for p in learner.parameters():
        # avg_norm.append(torch.linalg.norm(p.data))
        # print(torch.tensor(avg_norm))
        # avg_norm = torch.tensor(avg_norm).mean().item()
        #     print(f"Average inner weight norm: {avg_norm:.2f}")

        # Adapt the model on the support set
        for step in range(self._steps):
            # forward + backward + optimize
            outputs3d = learner(s_labels2d)
            # if torch.isnan(outputs3d).any():
            # print(f"Support outputs contains NaN!")
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            learner.adapt(support_loss, clip_grad_max_norm=clip_grad_norm)
            if msl:  # Multi-step loss
                q_outputs3d = learner(q_labels2d)
                query_loss += self._step_weights[step] * criterion(
                    q_outputs3d, q_labels3d
                )

        # with torch.no_grad():
        # avg_norm = []
        # for p in learner.parameters():
        # avg_norm.append(torch.linalg.norm(p.data))
        # print(torch.tensor(avg_norm))
        # avg_norm = torch.tensor(avg_norm).mean().item()
        # print(f"Average inner weight norm: {avg_norm:.2f}")

        # Evaluate the adapted model on the query set
        if not msl:
            q_outputs3d = learner(q_labels2d)
            # if torch.isnan(e_outputs3d).any():
            # print(f"Query outputs contains NaN!")
            query_loss = criterion(q_outputs3d, q_labels3d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        return self._training_step(
            meta_batch, learner, clip_grad_norm, compute, msl=False
        )


class Regular_CNNTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="18", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="34", pretrained=True)
        elif cnn_def == "resnet50":
            cnn = ResNet(model="50", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        super().__init__(
            cnn,
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        outputs3d, _ = self.model(inputs)
        loss = self.inner_criterion(outputs3d, labels3d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            outputs3d, _ = self.model(inputs)
            if compute == "mse":
                return F.mse_loss(outputs3d, labels3d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs3d, labels3d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")


class Regular_GraphUNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphUNetBatchNorm(),
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        _, labels2d, labels3d = batch
        if self._use_cuda:
            labels2d = labels2d.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        outputs3d = self.model(labels2d)
        loss = self.inner_criterion(outputs3d, labels3d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        _, labels2d, labels3d = batch
        if self._use_cuda:
            labels2d = labels2d.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            outputs3d = self.model(labels2d)
            if compute == "mse":
                return F.mse_loss(outputs3d, labels3d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs3d, labels3d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")


class Regular_GraphNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        resnet_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphNet(
                in_features=514, out_features=2
            ),  # 514 for the output features of resnet10
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        cnn_def = cnn_def.lower()
        if cnn_def == "resnet10":
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet18":
            cnn = ResNet(model="18", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="34", pretrained=True)
        elif cnn_def == "mobilenetv3-small":
            cnn = MobileNet(model="v3-small", pretrained=True)
        elif cnn_def == "mobilenetv3-large":
            cnn = MobileNet(model="v3-large", pretrained=True)
        else:
            raise ValueError(f"{cnn_def} is not a valid CNN definition!")
        self._resnet = cnn
        if use_cuda and torch.cuda.is_available():
            self._resnet = self._resnet.cuda()
            if resnet_path:
                print(f"[*] Loading ResNet state dict form {resnet_path}")
                load_state_dict(self._resnet, resnet_path)
            else:
                print("[!] ResNet is randomly initialized! It will be trained...")
            self._resnet = torch.nn.DataParallel(self._resnet, device_ids=gpu_numbers)
        self._resnet.eval()

    def _training_step(self, batch: tuple):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            points2D_init, features = self._resnet(inputs)
            features = features.unsqueeze(1).repeat(1, 29, 1)
            in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.model(in_features)
        loss = self.inner_criterion(points2D, labels2d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            points2D_init, features = self._resnet(inputs)
            features = features.unsqueeze(1).repeat(1, 29, 1)
            in_features = torch.cat([points2D_init, features], dim=2)
            points2D = self.model(in_features)
            if compute == "mse":
                return F.mse_loss(points2D, labels2d).detach()
            elif compute == "mae":
                return F.l1_loss(points2D, labels2d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")


class Regular_GraphNetwResNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        resnet_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphNetwResNet(cnn_def, resnet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        print("[*] Training ResNet with GraphNet end-to-end")

    def _training_step(self, batch: tuple):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)

        outputs2d_init, outputs2d = self.model(inputs)
        loss2d_init = self.inner_criterion(outputs2d_init, labels2d)
        loss2d = self.inner_criterion(outputs2d, labels2d)
        loss = loss2d_init + loss2d
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)

        with torch.no_grad():
            _, outputs2d = self.model(inputs)
            if compute == "mse":
                return F.mse_loss(outputs2d, labels2d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs2d, labels2d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")


class Regular_HOPENetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        resnet_path: str,
        graphnet_path: str,
        graphunet_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        inputs, labels2d, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)

        outputs2d_init, outputs2d, outputs3d = self.model(inputs)
        loss2d_init = self.inner_criterion(outputs2d_init, labels2d)
        loss2d = self.inner_criterion(outputs2d, labels2d)
        loss3d = self.inner_criterion(outputs3d, labels3d)
        loss = (
            (self._lambda1 * loss2d_init)
            + (self._lambda1 * loss2d)
            + (self._lambda2 * loss3d)
        )
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)

        with torch.no_grad():
            _, _, outputs3d = self.model(inputs)
            if compute == "mse":
                return F.mse_loss(outputs3d, labels3d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs3d, labels3d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")


class Regular_HOPENetTester(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        resnet_path: str,
        graphnet_path: str,
        graphunet_path: str,
        model_path: str = None,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        raise NotImplementedError

    def _testing_step(self, batch: tuple, compute="3d_ho_pcp"):
        inputs, labels2d, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)

        with torch.no_grad():
            _, outputs2d, outputs3d = self.model(inputs, gt_2d=None)
            if compute == "mse":
                return F.mse_loss(outputs3d, labels3d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs3d, labels3d).detach()
            elif compute == "mse2d":
                return F.mse_loss(outputs2d, labels2d).detach()
            elif compute == "mae2d":
                return F.l1_loss(outputs2d, labels2d).detach()
            else:
                raise NotImplementedError(f"No implementation for {compute}")

    def test(
        self,
        batch_size: int = 32,
        fast_lr: float = 0.01,
        meta_lr: float = None,
    ):
        import numpy as np

        if not self._model_path:
            print(f"[!] Testing a (partly) randomly initialized model!")
        else:
            print(f"[*] Restoring from checkpoint: {self._model_path}")
            checkpoint = torch.load(self._model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        (
            avg_mse_loss,
            avg_mae_loss,
            mse_losses,
            mse_losses2d,
            mae_losses,
            mae_losses2d,
        ) = (0.0, 0.0, [], [], [], [])
        err3d, err3d_hands, err2d_obj, err2d_init_obj, err2d_ho, err2d_init_ho = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        eps = 1e-5
        for batch in tqdm(self.dataset.test, dynamic_ncols=True):
            if self._exit:
                break
            inputs, labels2d, labels3d = batch
            if self._use_cuda:
                inputs = inputs.float().cuda(device=self._gpu_number)
                labels3d = labels3d.float().cuda(device=self._gpu_number)
                labels2d = labels2d.float().cuda(device=self._gpu_number)

            with torch.no_grad():
                for i, input_sample in enumerate(inputs):
                    outputs2d_init, outputs2d, outputs3d = self.model(
                        torch.unsqueeze(input_sample, dim=0)
                    )
                    err3d.append(
                        torch.mean(torch.linalg.norm((outputs3d - labels3d[i]), dim=0))
                    )
                    err3d_hands.append(
                        torch.mean(
                            torch.linalg.norm(
                                (outputs3d[:, :21, :] - labels3d[i, :21, :]), dim=0
                            )
                        )
                    )
                    err2d_obj.append(
                        torch.mean(
                            torch.linalg.norm(
                                (outputs2d[:, 21:, :] - labels2d[i, 21:, :]), dim=0
                            )
                        )
                    )
                    err2d_init_obj.append(
                        torch.mean(
                            torch.linalg.norm(
                                (outputs2d_init[:, 21:, :] - labels2d[i, 21:, :]), dim=0
                            )
                        )
                    )
                    err2d_ho.append(
                        torch.mean(torch.linalg.norm((outputs2d - labels2d[i]), dim=0))
                    )
                    err2d_init_ho.append(
                        torch.mean(
                            torch.linalg.norm((outputs2d_init - labels2d[i]), dim=0)
                        )
                    )
            mae_losses.append(self._testing_step(batch, compute="mae"))
            mse_losses.append(self._testing_step(batch, compute="mse"))
            mae_losses2d.append(self._testing_step(batch, compute="mae2d"))
            mse_losses2d.append(self._testing_step(batch, compute="mse2d"))
        avg_mse_loss = torch.Tensor(mse_losses).mean().item()
        avg_mse2d_loss = torch.Tensor(mse_losses2d).mean().item()
        avg_mae_loss = torch.Tensor(mae_losses).mean().item()
        avg_mae2d_loss = torch.Tensor(mae_losses2d).mean().item()
        print(f"[*] Average MSE test loss: {avg_mse_loss:.6f}")
        print(f"[*] Average MAE test loss: {avg_mae_loss:.6f}")
        print(f"[*] Average MSE 2D test loss: {avg_mse2d_loss:.6f}")
        print(f"[*] Average MAE 2D test loss: {avg_mae2d_loss:.6f}")

        max_thresh, thresh_step = 80, 5
        correct_ho_poses, correct_hand_poses = [], []
        for thresh in range(0, max_thresh, thresh_step):
            correct_ho_poses.append(
                len(torch.where(torch.tensor(err3d) <= thresh)[0])
                * 100.0
                / (len(err3d) + eps)
            )
            correct_hand_poses.append(
                len(torch.where(torch.tensor(err3d_hands) <= thresh)[0])
                * 100.0
                / (len(err3d_hands) + eps)
            )

        print(f"[*] Percentages of correct hand-object 3D poses: {correct_ho_poses}")
        print(f"[*] Percentages of correct hand 3D poses: {correct_hand_poses}")
        plt.plot(list(range(0, max_thresh, thresh_step)), correct_ho_poses)
        plt.xlabel("mm Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.grid(True, linestyle="dashed")
        plt.title("Percentage of Correct Hand-Object Poses (3D)")
        plt.savefig("ho_pcp3d.png")
        plt.clf()

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_hand_poses)
        plt.xlabel("mm Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Hand Poses (3D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("h_pcp3d.png")
        plt.clf()

        max_thresh, thresh_step = 50, 5
        correct_obj_poses2d, correct_obj_init_poses2d = [], []
        correct_ho_poses2d, correct_ho_init_poses2d = [], []
        for thresh in range(0, max_thresh, thresh_step):
            correct_obj_poses2d.append(
                len(torch.where(torch.tensor(err2d_obj) <= thresh)[0])
                * 100.0
                / (len(err2d_obj) + eps)
            )
            correct_obj_init_poses2d.append(
                len(torch.where(torch.tensor(err2d_init_obj) <= thresh)[0])
                * 100.0
                / (len(err2d_init_obj) + eps)
            )
            correct_ho_poses2d.append(
                len(torch.where(torch.tensor(err2d_ho) <= thresh)[0])
                * 100.0
                / (len(err2d_ho) + eps)
            )
            correct_ho_init_poses2d.append(
                len(torch.where(torch.tensor(err2d_init_ho) <= thresh)[0])
                * 100.0
                / (len(err2d_init_ho) + eps)
            )

        print(f"[*] Percentages of correct hand-object 2D poses: {correct_ho_poses2d}")
        print(f"[*] Percentages of correct object 2D poses: {correct_obj_poses2d}")
        plt.plot(list(range(0, max_thresh, thresh_step)), correct_obj_poses2d)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Object Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("o_pcp2d.png")
        plt.clf()

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_obj_init_poses2d)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Object Initial Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("o_pcp2d_init.png")
        plt.clf()

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_ho_poses2d)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Hand-Object Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("ho_pcp2d.png")
        plt.clf()

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_ho_init_poses2d)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Hand-Object Initial Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("ho_pcp2d_init.png")

        with open("test_results.pkl", "wb") as file:
            pickle.dump(
                {
                    "correct_ho_poses": correct_ho_poses,
                    "correct_hand_poses": correct_hand_poses,
                    "correct_ho_poses2d": correct_ho_poses2d,
                    "correct_ho_init_poses2d": correct_ho_init_poses2d,
                    "correct_obj_poses2d": correct_obj_poses2d,
                    "correct_obj_init_poses2d": correct_obj_init_poses2d,
                    "avg_mse": avg_mse_loss,
                    "avg_mae": avg_mae_loss,
                },
                file,
            )
