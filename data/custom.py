#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Custom dataset classes and interfaces.
"""

import os
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Union
from PIL import Image


class CustomDataset(TorchDataset):
    def __init__(
        self,
        samples: Union[dict, list],
        img_transform=None,
        kp2d_transform=None,
        object_as_task: bool = False,
        pin_memory=True,
        hand_only=True,
        to_ram_disk_fn=None,
    ):
        self._dim = 21 if hand_only else 29
        self._pin_memory = pin_memory
        self._to_ram_disk_fn = to_ram_disk_fn
        self._min_task_samples = float("+inf")
        self._img_transform = (
            img_transform if img_transform is not None else lambda i: i
        )
        self._kp2d_transform = (
            kp2d_transform if kp2d_transform is not None else lambda i: i
        )
        self.images, self.points2d, self.points3d, self.class_labels = self._load(
            samples, object_as_task
        )

    @property
    def min_task_samples(self):
        return self._min_task_samples

    def _load(
        self, samples: Union[Dict[int, List[tuple]], List[tuple]], object_as_task: bool
    ) -> tuple:
        images, points2d, points3d = [], [], []
        labels, i = {}, 0

        def load_sample(img_path, p_2d, p_3d):
            if self._to_ram_disk_fn is not None:
                img_path = self._to_ram_disk_fn(img_path)
            images.append(img_path)
            p_3d = p_3d[: self._dim] - p_3d[0, :]  # Root aligned
            if self._pin_memory:
                p_2d.pin_memory()
                p_3d.pin_memory()
            points2d.append(p_2d)
            points3d.append(p_3d)

        if object_as_task:
            n = 0
            for k, v in samples.items():
                for img_path, p_2d, p_3d in v:
                    load_sample(img_path, p_2d, p_3d)
                    labels[i] = k
                    i += 1
                if len(v) < self._min_task_samples:
                    self._min_task_samples = len(v)
                n += 1
        else:
            for img_path, p_2d, p_3d in samples:
                load_sample(img_path, p_2d, p_3d)
        return images, points2d, points3d, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self._img_transform(Image.open(self.images[index]))
        labels_2d, labels_3d = (
            self._kp2d_transform(self.points2d[index]),
            self.points3d[index],
        )
        return img, labels_2d, labels_3d
