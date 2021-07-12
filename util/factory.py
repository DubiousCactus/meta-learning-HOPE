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
from data.dataset.ho3d import HO3DTaskLoader
from hydra.utils import to_absolute_path
from algorithm.wrappers import (
    Regular_GraphUNetTrainer,
    Regular_GraphNetTrainer,
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
        augment_fphad: bool = False,
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
            kargs = {'augment_2d': augment_fphad}
        elif dataset == "ho3d":
            datasetClass = HO3DTaskLoader
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
            use_cuda=use_cuda,
            gpu_number=gpu_numbers[0],
            **kargs
        )


class AlgorithmFactory:
    @abstractmethod
    def make_training_algorithm(
        config,
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
        args, kargs = [], {}
        trainer = None
        if algorithm in ["maml", "fomaml"]:
            if model_def == "hopenet":
                trainer = MAML_HOPETrainer
            elif model_def == "resnet":
                trainer = MAML_ResnetTrainer
            elif model_def == "graphunet":
                trainer = MAML_GraphUNetTrainer
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
            args = [k_shots, n_queries, inner_steps]
            kargs = {'first_order': algorithm == "fomaml"}
        elif algorithm == "regular":
            if model_def == "hopenet":
                raise Exception(f"No training algorithm found for model {model_def}")
            elif model_def == "resnet":
                trainer = Regular_ResnetTrainer
            elif model_def == "graphunet":
                trainer = Regular_GraphUNetTrainer
            elif model_def == "graphnet":
                trainer = Regular_GraphNetTrainer
                resnet_path = config.experiment.resnet_model_path
                args = [to_absolute_path(resnet_path) if resnet_path else None]
            else:
                raise Exception(f"No training algorithm found for model {model_def}")
        else:
            raise Exception(f"No training algorithm found: {algorithm}")
        return trainer(
                dataset,
                ckpt_path,
                *args,
                model_path=model_path,
                use_cuda=use_cuda,
                gpu_numbers=gpu_numbers,
                **kargs
                )
