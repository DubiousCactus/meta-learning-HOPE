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

from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Dict, List, Union
from torch.autograd import Variable
from torch import Tensor
from PIL import Image

import torch


class CustomDataset(TorchDataset):
    def __init__(
        self,
        samples: any,
        transform=None,
        object_as_task: bool = False,
        pin_memory=True,
    ):
        self._pin_memory = pin_memory
        self.transform = transform if transform is not None else lambda i: i
        self.images, self.points2d, self.points3d, self.class_labels = self._load(
            samples, object_as_task
        )

    def _load(
        self, samples: Union[Dict[int, List[tuple]], List[tuple]], object_as_task: bool
    ) -> tuple:
        def load_sample(img_path, p_2d, p_3d) -> tuple:
            img = img_path
            if self._pin_memory:
                p_2d.pin_memory()
                p_3d.pin_memory()
            return img, p_2d, p_3d

        images, points2d, points3d = [], [], []
        labels, i = {}, 0
        if object_as_task:
            for k, v in samples.items():
                for img_path, p_2d, p_3d in v:
                    img, p_2d, p_3d = load_sample(img_path, p_2d, p_3d)
                    images.append(img)
                    points2d.append(p_2d)
                    points3d.append(p_3d)
                    labels[i] = k
                    i += 1
        else:
            for img_path, p_2d, p_3d in samples:
                img, p_2d, p_3d = load_sample(img_path, p_2d, p_3d)
                images.append(img)
                points2d.append(p_2d)
                points3d.append(p_3d)
        return images, points2d, points3d, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.images[index]))
        labels_2d, labels_3d = (
            self.points2d[index],
            self.points3d[index],
        )
        return img, labels_2d, labels_3d
