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

from data.dataset.obman import ObManTaskLoader
from data.dataset.fphad import FPHADTaskLoader
from abc import abstractmethod

import os


class DatasetFactory:
    # TODO: Get this out of here
    _shapenet_root = "/home/cactus/Code/ShapeNetCore.v2/"

    @abstractmethod
    def make_data_loader(
        dataset: str,
        dataset_root: str,
        batch_size: int,
        test: bool,
        object_as_task: bool,
        k_shots: int,
    ):
        if not os.path.isdir(dataset_root):
            print(f"[!] {dataset_root} is not a valid directory!")
            exit(1)
        dataset = dataset.lower()
        if dataset == "obman":
            return ObManTaskLoader(
                dataset_root,
                DatasetFactory._shapenet_root,
                batch_size,
                k_shots,
                test=test,
                object_as_task=object_as_task,
            )
        elif dataset == "fphad":
            return FPHADTaskLoader(
                dataset_root,
                batch_size,
                k_shots,
                test=test,
                object_as_task=object_as_task,
            )
        elif dataset == "ho3d":
            raise NotImplementedError("HO-3D Dataset Loader not implemented!")
        else:
            raise NotImplementedError(f"{dataset} Dataset Loader not implemented!")
