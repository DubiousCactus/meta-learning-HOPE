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

from data.dataset.base import BaseDatasetTaskLoader
from data.dataset.obman import ObManTaskLoader
from data.dataset.fphad import FPHADTaskLoader
from algorithm.wrappers import (
    Regular_GraphUNetTrainer,
    Regular_ResnetTrainer,
    MAML_GraphUNetTrainer,
    MAML_ResnetTrainer,
    MAML_HOPETrainer,
)
from algorithm.base import BaseTrainer
from abc import abstractmethod
from typing import List

import os


class DatasetFactory:
    @abstractmethod
    def make_data_loader(
        shapenet_root,
        dataset: str,
        dataset_root: str,
        batch_size: int,
        test: bool,
        k_shots: int,
        n_querries: int,
        object_as_task: bool = False,
        use_cuda: bool = True,
        gpu_numbers: List = [0],
        augment_fphad: bool = True,
    ):
        if not os.path.isdir(dataset_root):
            print(f"[!] {dataset_root} is not a valid directory!")
            exit(1)
        dataset = dataset.lower()
        print(f"[*] Loading dataset: {dataset}")
        if dataset == "obman":
            return ObManTaskLoader(
                dataset_root,
                shapenet_root,
                batch_size,
                k_shots,
                n_querries,
                test=test,
                object_as_task=object_as_task,
                use_cuda=use_cuda,
                gpu_number=gpu_numbers[0],
            )
        elif dataset == "fphad":
            return FPHADTaskLoader(
                dataset_root,
                batch_size,
                k_shots,
                n_querries,
                test=test,
                object_as_task=object_as_task,
                use_cuda=use_cuda,
                gpu_number=gpu_numbers[0],
                augment_2d=augment_fphad,
            )
        elif dataset == "ho3d":
            raise NotImplementedError("HO-3D Dataset Loader not implemented!")
        else:
            raise NotImplementedError(f"{dataset} Dataset Loader not implemented!")


class AlgorithmFactory:
    @abstractmethod
    def make_training_algorithm(
        algorithm: str,
        model_def: str,
        dataset: BaseDatasetTaskLoader,
        k_shots: int,
        n_queries: int,
        inner_steps: int,
        ckpt_path: str,
        model_path: str = None,
        use_cuda: bool = True,
        gpu_numbers: List = [0],
    ) -> BaseTrainer:
        trainer = lambda x: Exception()
        algorithm = algorithm.lower()
        model_def = model_def.lower()
        print(f"[*] Loading training algorithm: {algorithm}")
        print(f"[*] Loading model definition: {model_def}")
        if algorithm in ["maml", "fomaml"]:
            if model_def == "hopenet":
                trainer = MAML_HOPETrainer
            elif model_def == "resnet":
                trainer = MAML_ResnetTrainer
            elif model_def == "graphunet":
                trainer = MAML_GraphUNetTrainer
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
            return trainer(
                dataset,
                ckpt_path,
                k_shots,
                n_queries,
                inner_steps,
                model_path=model_path,
                use_cuda=use_cuda,
                gpu_numbers=gpu_numbers,
                first_order=(algorithm == "fomaml"),
            )
        elif algorithm == "regular":
            if model_def == "hopenet":
                raise Exception(f"No training algorithm found for model {model_def}")
            elif model_def == "resnet":
                trainer = Regular_ResnetTrainer
            elif model_def == "graphunet":
                trainer = Regular_GraphUNetTrainer
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
            return trainer(
                dataset,
                ckpt_path,
                model_path=model_path,
                use_cuda=use_cuda,
                gpu_numbers=gpu_numbers,
            )

        else:
            raise Exception(f"No training algorithm found: {algorithm}")
        return trainer
