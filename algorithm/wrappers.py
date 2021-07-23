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

from model.cnn import ResNet, MobileNet
from model.hopenet import HOPENet

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
        n_querries,
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
            n_querries,
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
        n_querries,
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
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="10", pretrained=True)
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
            n_querries,
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
        n_querries,
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
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="10", pretrained=True)
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
            n_querries,
            inner_steps,
            model_path=model_path,
            first_order=first_order,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(
        self, batch: MetaBatch, head, features, clip_grad_norm=None, compute="mse"
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

        s_inputs = features(s_inputs)
        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs2d_init, _ = head(s_inputs)
            if torch.isnan(outputs2d_init).any():
                print(f"Support outputs contains NaN!")
            support_loss = self.inner_criterion(outputs2d_init, s_labels2d)
            head.adapt(support_loss, clip_grad_max_norm=clip_grad_norm)

        # Evaluate the adapted model on the query set
        q_inputs = features(q_inputs)
        e_outputs2d_init, _ = head(q_inputs)
        if torch.isnan(e_outputs2d_init).any():
            print(f"Query outputs contains NaN!")
        query_loss = criterion(e_outputs2d_init, q_labels2d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, head, features, clip_grad_norm=None, compute="mse"
    ):
        return self._training_step(meta_batch, head, features, clip_grad_norm, compute)


class MAML_GraphUNetTrainer(MAMLTrainer):
    def __init__(
        self,
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
        super().__init__(
            "graphunet",
            dataset,
            checkpoint_path,
            k_shots,
            n_querries,
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
        _, s_labels2d, s_labels3d = batch.support
        _, q_labels2d, q_labels3d = batch.query
        if self._use_cuda:
            s_labels2d = s_labels2d.float().cuda(device=self._gpu_number)
            s_labels3d = s_labels3d.float().cuda(device=self._gpu_number)
            q_labels2d = q_labels2d.float().cuda(device=self._gpu_number)
            q_labels3d = q_labels3d.float().cuda(device=self._gpu_number)

        # Adapt the model on the support set
        for _ in range(self._steps):
            # forward + backward + optimize
            outputs3d = learner(s_labels2d)
            if torch.isnan(outputs3d).any():
                print(f"Support outputs contains NaN!")
            support_loss = self.inner_criterion(outputs3d, s_labels3d)
            learner.adapt(support_loss, clip_grad_max_norm=clip_grad_norm)

        # Evaluate the adapted model on the query set
        e_outputs3d = learner(q_labels2d)
        if torch.isnan(e_outputs3d).any():
            print(f"Query outputs contains NaN!")
        query_loss = criterion(e_outputs3d, q_labels3d)
        return query_loss

    def _testing_step(
        self, meta_batch: MetaBatch, learner, clip_grad_norm=None, compute="mse"
    ):
        return self._training_step(meta_batch, learner, clip_grad_norm, compute)


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
            cnn = ResNet(model="10", pretrained=True)
        elif cnn_def == "resnet34":
            cnn = ResNet(model="10", pretrained=True)
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
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        outputs2d_init, _ = self.model(inputs)
        loss = self.inner_criterion(outputs2d_init, labels2d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, loss_fn=None):
        criterion = self.inner_criterion if not loss_fn else loss_fn
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            outputs2d_init, _ = self.model(inputs)
            return criterion(outputs2d_init, labels2d).detach()


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
            "graphunet",
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

    def _testing_step(self, batch: tuple, loss_fn=None):
        criterion = self.inner_criterion if not loss_fn else loss_fn
        _, labels2d, labels3d = batch
        if self._use_cuda:
            labels2d = labels2d.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            outputs3d = self.model(labels2d)
            return criterion(outputs3d, labels3d).detach()


class Regular_GraphNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
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
        self._resnet = select_model("resnet10")
        if use_cuda and torch.cuda.is_available():
            self._resnet = self._resnet.cuda()
            self._resnet = torch.nn.DataParallel(self._resnet, device_ids=gpu_numbers)
        if resnet_path:
            print(f"[*] Loading ResNet state dict form {resnet_path}")
            ckpt = torch.load(resnet_path)
            self._resnet.load_state_dict(ckpt["model_state_dict"])
        else:
            print("[!] ResNet is randomly initialized!")
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

    def _testing_step(self, batch: tuple, loss_fn=None):
        criterion = self.inner_criterion if not loss_fn else loss_fn
        inputs, labels2d, _ = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            points2D_init, features = self._resnet(inputs)
            features = features.unsqueeze(1).repeat(1, 29, 1)
            in_features = torch.cat([points2D_init, features], dim=2)
            points2D = self.model(in_features)
            return criterion(points2D, labels2d).detach()


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
            (self._lambda1) * loss2d_init
            + (self._lambda1) * loss2d
            + (self._lambda2) * loss3d
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

    def _testing_step(self, batch: tuple, compute="3d_ho_pcp", threshold=0):
        inputs, labels2d, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
            labels2d = labels2d.float().cuda(device=self._gpu_number)

        with torch.no_grad():
            outputs2d_init, outputs2d, outputs3d = self.model(inputs)
            if compute == "mse":
                return F.mse_loss(outputs3d, labels3d).detach()
            elif compute == "mae":
                return F.l1_loss(outputs3d, labels3d).detach()
            elif compute == "3d_ho_pcp":
                # Percentage of Correct (hand-object) Poses (3D-PCP for hand-object):
                mean_norms = torch.mean(
                    torch.norm((outputs3d - labels3d), dim=2),
                    dim=1,
                )
                return int(
                    torch.where(mean_norms < threshold, 1, 0).count_nonzero().item()
                )
            elif compute == "3d_hand_pcp":
                # Percentage of Correct (hand) Poses (3D-PCP for hands):
                mean_norms = torch.mean(
                    torch.norm((outputs3d[:, :21, :] - labels3d[:, :21, :]), dim=2),
                    dim=1,
                )
                return int(
                    torch.where(mean_norms < threshold, 1, 0).count_nonzero().item()
                )
            elif compute == "2d_obj_pcp":
                # Percentage of Correct (object) Poses (2D-PCP for objects, after graph
                # convolutions):
                mean_norms = torch.mean(
                    torch.norm((outputs2d[:, 21:, :] - labels2d[:, 21:, :]), dim=2),
                    dim=1,
                )
                return int(
                    torch.where(mean_norms < threshold, 1, 0).count_nonzero().item()
                )
            elif compute == "2dinit_obj_pcp":
                # Percentage of Correct (object) Poses (2D-PCP for objects, after resnet):
                mean_norms = torch.mean(
                    torch.norm(
                        (outputs2d_init[:, 21:, :] - labels2d[:, 21:, :]), dim=2
                    ),
                    dim=1,
                )
                return int(
                    torch.where(mean_norms < threshold, 1, 0).count_nonzero().item()
                )
            else:
                raise NotImplementedError(f"No implementation for {compute}")

    def test(
        self,
        batch_size: int = 32,
        fast_lr: float = 0.01,
        meta_lr: float = None,
    ):
        if not self._model_path:
            print(f"[!] Testing a randomly initialized model!")
        else:
            print(f"[*] Restoring from checkpoint: {self._model_path}")
            checkpoint = torch.load(self._model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        avg_mse_loss, avg_mae_loss, mse_losses, mae_losses = 0.0, 0.0, [], []
        for batch in tqdm(self.dataset.test, dynamic_ncols=True):
            if self._exit:
                return
            mae_losses.append(self._testing_step(batch, compute="mae"))
            mse_losses.append(self._testing_step(batch, compute="mse"))
        avg_mse_loss = torch.Tensor(mse_losses).mean().item()
        avg_mae_loss = torch.Tensor(mae_losses).mean().item()
        print(f"[*] Average MSE test loss: {avg_mse_loss:.6f}")
        print(f"[*] Average MAE test loss: {avg_mae_loss:.6f}")

        # Code tested and correct
        max_thresh, thresh_step = 100, 5
        (correct_ho_poses, correct_hand_poses,) = (
            [0] * (max_thresh // thresh_step),
            [0] * (max_thresh // thresh_step),
        )
        for i, thresh in tqdm(
            enumerate(range(0, max_thresh, thresh_step)),
            total=max_thresh // thresh_step,
            dynamic_ncols=True,
        ):
            for batch in self.dataset.test:
                if self._exit:
                    return
                correct_ho_poses[i] += self._testing_step(
                    batch, compute="3d_ho_pcp", threshold=thresh
                )
                correct_hand_poses[i] += self._testing_step(
                    batch, compute="3d_hand_pcp", threshold=thresh
                )

            for correct_poses in [
                correct_ho_poses,
                correct_hand_poses,
            ]:
                correct_poses[i] = int(
                    (correct_poses[i] * 100) / len(self.dataset.test.dataset)
                )

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

        max_thresh, thresh_step = 100, 5
        (correct_obj_poses, correct_obj_init_poses,) = (
            [0] * (max_thresh // thresh_step),
            [0] * (max_thresh // thresh_step),
        )
        for i, thresh in tqdm(
            enumerate(range(0, max_thresh, thresh_step)),
            total=max_thresh // thresh_step,
            dynamic_ncols=True,
        ):
            for batch in self.dataset.test:
                if self._exit:
                    return
                correct_obj_poses[i] += self._testing_step(
                    batch, compute="2d_obj_pcp", threshold=thresh
                )
                correct_obj_init_poses[i] += self._testing_step(
                    batch, compute="2dinit_obj_pcp", threshold=thresh
                )

            for correct_poses in [
                correct_obj_poses,
                correct_obj_init_poses,
            ]:
                correct_poses[i] = int(
                    (correct_poses[i] * 100) / len(self.dataset.test.dataset)
                )

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_obj_poses)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Object Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("o_pcp2d.png")
        plt.clf()

        plt.plot(list(range(0, max_thresh, thresh_step)), correct_obj_init_poses)
        plt.xlabel("pixel Threshold")
        plt.ylabel("Percentage of Correct Poses")
        plt.title("Percentage of Correct Object Initial Poses (2D)")
        plt.grid(True, linestyle="dashed")
        plt.savefig("o_pcp2d_init.png")

        with open("test_results.pkl", "wb") as file:
            pickle.dump(
                {
                    "correct_ho_poses": correct_ho_poses,
                    "correct_hand_poses": correct_hand_poses,
                    "correct_obj_poses": correct_obj_poses,
                    "correct_obj_init_poses": correct_obj_init_poses,
                    "avg_mse": avg_mse_loss,
                    "avg_mae": avg_mae_loss,
                },
                file,
            )
