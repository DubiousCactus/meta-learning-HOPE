#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Load datasets (ObMan, FPHAD, HO-3D) as sets of tasks, where each task is a set of manipulation
frames for one object class.
"""

import torchvision.transforms as transforms
import learn2learn as l2l

from torch.utils.data import DataLoader
from data.custom import CustomDataset
from typing import Union
from abc import ABC


class BaseDatasetTaskLoader(ABC):
    _img_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        n_queries: int,
        test: bool,
        object_as_task: bool,
        normalize_keypoints: bool,
        use_cuda: bool,
        gpu_number: int,
        auto_load: bool = True,
    ):
        self._root = root
        self._batch_size = batch_size
        self.k_shots = k_shots
        self.n_queries = n_queries
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number
        self.train, self.val, self.test = None, None, None
        if auto_load:
            if test:
                self.test = self._load(
                    object_as_task, "test", False, normalize_keypoints
                )
            else:
                print("LOADING")
                self.train, self.val = self._load(
                    object_as_task, "train", True, normalize_keypoints
                ), self._load(object_as_task, "val", False, normalize_keypoints)

    def _make_dataset(
        self, root, object_as_task=False, normalize_keypoints=False
    ) -> CustomDataset:
        raise NotImplementedError

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool, normalize_keypoints: bool
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        raise NotImplementedError
