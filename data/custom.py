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

from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from typing import Tuple, Dict, List
from torch import Tensor
from PIL import Image


class CustomDataset(TorchDataset):
    def __init__(self, samples: List[tuple], transform=None, lazy=True):
        self._lazy = lazy
        self.transform = transform if transform is not None else lambda i: i
        self.images, self.points2d, self.points3d = self._load(samples)

    def _load(self, samples: List[tuple]) -> tuple:
        images, points2d, points3d = [], [], []
        for img_path, p_2d, p_3d in samples:
            images.append(
                img_path if self._lazy else self.transform(Image.open(img_path))
            )
            points2d.append(p_2d)
            points3d.append(p_3d)
        return images, points2d, points3d

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = (
            self.transform(Image.open(self.images[index]))
            if self._lazy
            else self.images[index]
        )
        # TODO: Return a Pillow image to test the batch!
        assert (
            type(img) is Tensor
        ), "Image is not a tensor! Perhaps you forgot a transform?"
        return img[:3], self.points2d[index], self.points3d[index]


class CustomTaskDataset(TorchDataset):
    def __init__(self, image_paths: Dict[int, List[tuple]], transform=None, lazy=True):
        self._lazy = lazy
        self.transform = transform if transform is not None else lambda i: i
        self.images, self.class_labels, self.points2d, self.points3d = self._load(
            image_paths
        )

    def _load(self, image_paths: Dict[int, List[tuple]]) -> tuple:
        images, labels, i = [], {}, 0
        points2d, points3d = [], []
        for k, v in image_paths.items():
            for img_path, c_2d, c_3d in v:
                images.append(
                    img_path if self._lazy else self.transform(Image.open(img_path))
                )
                points2d.append(c_2d)
                points3d.append(c_3d)
                labels[i] = k
                i += 1
        return images, labels, points2d, points3d

    def __getitem__(self, index):
        img = (
            self.transform(Image.open(self.images[index]))
            if self._lazy
            else self.images[index]
        )
        # TODO: Return a Pillow image to test the batch!
        assert (
            type(img) is torch.Tensor
        ), "Image is not a tensor! Perhaps you forgot a transform?"
        return img[:3], self.points2d[index], self.points3d[index]

    def __len__(self):
        return len(self.images)


class CompatDataLoader(TorchDataLoader):
    """
    Simply extend PyTorch's DataLoader with a sample() function to make it seemlessly compatible
    with learn2learn's TaskDataset or MetaDataset, so that the interface is the same.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=2,
        persistent_workers=False,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.dataset_iter = iter(self.dataset)

    def sample(self):
        for indices in self.batch_sampler:
            yield self.collate_fn([next(self.dataset_iter) for _ in indices])


