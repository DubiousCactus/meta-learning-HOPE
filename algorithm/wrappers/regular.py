#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Regular training wrappers
"""

from util.utils import load_state_dict, plot_3D_pred_gt, select_cnn_model
from data.dataset.base import BaseDatasetTaskLoader
from model.hopenet import HOPENet, GraphNetwResNet
from model.graphnet import GraphUNetBatchNorm
from algorithm.regular import RegularTrainer
from HOPE.models.graphunet import GraphNet

from typing import List
from tqdm import tqdm

import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


class Regular_CNNTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        cnn_def: str,
        model_path: str = None,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            select_cnn_model(cnn_def, hand_only),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )

    def _training_step(self, batch: tuple):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        joints, _ = self.model(inputs)
        joints -= joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1)  # Root alignment
        loss = self.inner_criterion(joints, labels3d)
        loss.backward()
        return loss.detach()

    def _testing_step(self, batch: tuple, compute="mse"):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
            labels3d = labels3d.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            joints, _ = self.model(inputs)
            joints -= (
                joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1)
            )  # Root alignment
            res = None
            if type(compute) is str:
                """
                This will be used when validating.
                """
                if compute == "mse":
                    res = self.inner_criterion(joints, labels3d).detach()
                elif compute == "mae":
                    res = F.l1_loss(joints, labels3d).detach()
                elif compute == "mpjpe":
                    # Hand-pose only
                    # Batched vector norm for row-wise elements
                    return (
                        torch.linalg.norm(
                            joints[:, :self._dim, :] - labels3d[:, :self._dim, :], dim=2
                        )
                        .detach()
                        .mean()
                    )
            elif type(compute) is list:
                """
                This will be used when testing.
                """
                res = {}
                for metric in compute:
                    if metric == "mse":
                        res[metric] = (self.inner_criterion(joints, labels3d).detach())
                    elif metric == "mae":
                        res[metric] = (F.l1_loss(joints, labels3d).detach())
                    elif metric == "pjpe":
                        # Hand-pose only
                        # Batched vector norm for row-wise elements
                        res[metric] = (
                            torch.linalg.norm(
                                joints[:, :self._dim, :] - labels3d[:, :self._dim, :], dim=2
                            )
                            .detach()
                        )
                    elif metric == "pcpe":
                        # Object-pose only
                        # Batched vector norm for row-wise elements
                        res[metric] = (
                            torch.linalg.norm(
                                joints[:, self._dim:, :] - labels3d[:, self._dim:, :], dim=2
                            )
                            .detach()
                        )
        assert res is not None, f"{compute} is not a valid metric!"
        return res

    def _testing_step_vis(self, batch: tuple):
        inputs, _, labels3d = batch
        if self._use_cuda:
            inputs = inputs.float().cuda(device=self._gpu_number)
        with torch.no_grad():
            joints, _ = self.model(inputs)
            joints -= (
                joints[:, 0, :].unsqueeze(dim=1).expand(-1, self._dim, -1)
            )  # Root alignment
            mean, std = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32
            ), torch.tensor([0.221, 0.224, 0.225], dtype=torch.float32)
            unnormalize = transforms.Normalize(
                mean=(-mean / std).tolist(), std=(1.0 / std).tolist()
            )
            unnormalized_img = unnormalize(inputs[0])
            npimg = (
                (unnormalized_img * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )
            plot_3D_pred_gt(joints[0].cpu(), npimg, labels3d[0].cpu())


class Regular_GraphUNetTrainer(RegularTrainer):
    def __init__(
        self,
        dataset: BaseDatasetTaskLoader,
        checkpoint_path: str,
        model_path: str = None,
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphUNetBatchNorm(),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
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
        hand_only: bool = True,
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
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
        )
        self._resnet = select_cnn_model(cnn_def, hand_only)
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
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            GraphNetwResNet(cnn_def, resnet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
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
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
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
        hand_only: bool = True,
        use_cuda: int = False,
        gpu_numbers: List = [0],
    ):
        super().__init__(
            HOPENet(cnn_def, resnet_path, graphnet_path, graphunet_path),
            dataset,
            checkpoint_path,
            model_path=model_path,
            hand_only=hand_only,
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
                                (outputs3d[:, :self._dim, :] - labels3d[i, :self._dim, :]), dim=0
                            )
                        )
                    )
                    err2d_obj.append(
                        torch.mean(
                            torch.linalg.norm(
                                (outputs2d[:, self._dim:, :] - labels2d[i, self._dim:, :]), dim=0
                            )
                        )
                    )
                    err2d_init_obj.append(
                        torch.mean(
                            torch.linalg.norm(
                                (outputs2d_init[:, self._dim:, :] - labels2d[i, self._dim:, :]), dim=0
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
