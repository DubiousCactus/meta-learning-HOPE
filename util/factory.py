#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utility classes.
"""

from data.dataset.dex_ycb import DexYCBDatasetTaskLoader
from data.dataset.base import BaseDatasetTaskLoader
from data.dataset.obman import ObManTaskLoader
from data.dataset.fphad import FPHADTaskLoader
from data.dataset.ho3d import HO3DTaskLoader
from algorithm.wrappers.anil import ANIL_CNNTrainer
from algorithm.wrappers.maml import (
    MAML_CNNTrainer,
)
from algorithm.wrappers.regular import (
    Regular_CNNTrainer,
)
from algorithm.base import BaseTrainer
from abc import abstractmethod
from typing import List

import os


class DatasetFactory:
    @abstractmethod
    def make_data_loader(
        config,
        shapenet_root,
        dataset: str,
        dataset_root: str,
        batch_size: int,
        test: bool,
        k_shots: int,
        n_querries: int,
        hand_only: bool,
        object_as_task: bool = False,
        normalize_keypoints: bool = False,
        use_cuda: bool = True,
        gpu_numbers: List = [0],
        augment_fphad: bool = False,
        auto_load: bool = True,
    ) -> BaseDatasetTaskLoader:
        if not os.path.isdir(dataset_root):
            print(f"[!] {dataset_root} is not a valid directory!")
            exit(1)
        dataset = dataset.lower()
        print(f"[*] Loading dataset: {dataset}")
        args, kargs = [], {}
        if dataset == "obman":
            datasetClass = ObManTaskLoader
            args = [shapenet_root]
        elif dataset == "fphad":
            datasetClass = FPHADTaskLoader
            kargs = {"augment_2d": augment_fphad}
        elif dataset == "ho3d":
            datasetClass = HO3DTaskLoader
            kargs = {"hold_out": config.experiment.hold_out}
        elif dataset == "dexycb":
            datasetClass = DexYCBDatasetTaskLoader
            kargs = {
                "hold_out": config.experiment.hold_out,
                "test_objects": config.experiment.test_objects,
                "seed_factor": config.experiment.seed_factor,
                "auto_load": auto_load,
                "tiny": config.experiment.tiny,
            }
        else:
            raise NotImplementedError(f"{dataset} Dataset Loader not implemented!")
        return datasetClass(
            dataset_root,
            batch_size,
            k_shots,
            n_querries,
            *args,
            test=test,
            object_as_task=object_as_task,
            normalize_keypoints=normalize_keypoints,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_number=gpu_numbers[0],
            **kargs,
        )


class AlgorithmFactory:
    @abstractmethod
    def make_training_algorithm(
        config,
        algorithm: str,
        model_def: str,
        cnn_def: str,
        dataset: BaseDatasetTaskLoader,
        k_shots: int,
        n_queries: int,
        inner_steps: int,
        ckpt_path: str,
        test_mode: bool,
        hand_only: bool,
        model_path: str = None,
        use_cuda: bool = True,
        gpu_numbers: List = [0],
    ) -> BaseTrainer:
        trainer = lambda x: Exception()
        print(f"[*] Loading training algorithm: {algorithm}")
        print(f"[*] Loading model definition: {model_def}")
        algorithm = algorithm.lower()
        model_def = model_def.lower()
        args, kargs = [], {}
        trainer = None
        if algorithm in ["maml", "fomaml"]:
            args: List = [k_shots, n_queries, inner_steps]
            if "resnet" in model_def or "mobilenet" in model_def:
                trainer = MAML_CNNTrainer
                args += [model_def]
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
            kargs = {
                "first_order": algorithm == "fomaml",
                "multi_step_loss": config.experiment.multi_step_loss,
                "msl_num_epochs": config.experiment.msl_num_epochs,
                "task_aug": config.experiment.task_aug,
            }
        elif algorithm in ["anil", "foanil"]:
            args: List = [k_shots, n_queries, inner_steps]
            if "resnet" in model_def or "mobilenet" in model_def:
                trainer = ANIL_CNNTrainer
                args += [model_def]
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
            kargs = {
                "first_order": algorithm == "foanil",
                "multi_step_loss": config.experiment.multi_step_loss,
                "msl_num_epochs": config.experiment.msl_num_epochs,
                "beta": config.experiment.beta,
                "meta_reg": config.experiment.meta_reg,
                "task_aug": config.experiment.task_aug,
                "reg_bottleneck_dim": config.experiment.reg_bottleneck_dim,
            }
        elif algorithm == "regular":
            if "resnet" in model_def or "mobilenet" in model_def:
                trainer = Regular_CNNTrainer
                args = [model_def]
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
        else:
            raise Exception(f"No training algorithm found: {algorithm}")
        return trainer(
            dataset,
            ckpt_path,
            *args,
            model_path=model_path,
            hand_only=hand_only,
            use_cuda=use_cuda,
            gpu_numbers=gpu_numbers,
            **kargs,
        )
