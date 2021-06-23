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
        lazy=True,
        use_cuda=True,
        gpu_number=0,
        pin_memory=False,
    ):
        self._gpu_number = gpu_number
        self._use_cuda = use_cuda
        self._lazy = lazy
        self._pin_memory = pin_memory
        self.transform = transform if transform is not None else lambda i: i
        self.images, self.points2d, self.points3d, self.class_labels = self._load(
            samples, object_as_task
        )
        if not self._lazy:
            self.images = [
                self._preprocess(img, image_type=True) for img in self.images
            ]
            self.points2d = [self._preprocess(p) for p in self.points2d]
            self.points3d = [self._preprocess(p) for p in self.points3d]

    def _preprocess(self, v, image_type=False) -> List:
        v = Variable(v[:3]) if image_type else Variable(v)
        if self._use_cuda and torch.cuda.is_available():
            v = v.float().cuda(device=self._gpu_number)
        return v

    def _load(
        self, samples: Union[Dict[int, List[tuple]], List[tuple]], object_as_task: bool
    ) -> tuple:
        def load_sample(img_path, p_2d, p_3d) -> tuple:
            img = img_path
            if self._pin_memory:
                if not self._lazy:
                    img = self.transform(Image.open(img_path))
                    img.pin_memory()
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
        if self._lazy:
            img = self.transform(Image.open(self.images[index]))
            img, labels_2d, labels_3d = (
                self._preprocess(img, image_type=True),
                self._preprocess(self.points2d[index]),
                self._preprocess(self.points3d[index]),
            )
        else:
            img = self.images[index]
            labels_2d = self.points2d[index]
            labels_3d = self.points3d[index]

        assert (
            type(img) is Tensor
        ), "Image is not a tensor! Perhaps you forgot a transform?"
        return img, labels_2d, labels_3d


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
        use_cuda=True,
        gpu_number=0,
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
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number

    def _get_batch(self):
        def to_cuda(batch):
            if self._use_cuda and torch.cuda.is_available():
                return [e.cuda(device=self._gpu_number) for e in batch]

        for indices in self.batch_sampler:
            yield self.collate_fn(
                [
                    to_cuda(
                        [
                            Variable(Tensor(e)).float()
                            if (type(e) is not Tensor or not e.is_cuda)
                            else e
                            for e in next(self.dataset_iter)
                        ]
                    )
                    for _ in indices
                ]
            )

    def sample(self):
        return next(self._get_batch())
