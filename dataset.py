#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Load datasets (ObMan, FPHAD, HO-3D) as sets of tasks, where each task is a set of manipulation
frames for one unique object.
"""

import torchvision.transforms as transforms
import torch
import os

from HOPE.utils.dataset import Dataset
from abc import abstractmethod, ABC


class BaseDatasetTaskLoader(ABC):
    _transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def __init__(self, root: str, batch_size: int, test: bool, object_as_task: bool):
        self._root = root
        self._batch_size = batch_size
        self.train, self.val, self.test = None, None, None
        if test:
            self.test = self._load_test(object_as_task)
        else:
            self.train, self.val = self._load_train_val(object_as_task)

    def _load_test(self, object_as_task: bool):
        raise NotImplementedError

    def _load_train_val(self, object_as_task: bool):
        raise NotImplementedError


class DatasetFactory:
    @abstractmethod
    def make_data_loader(dataset: str, dataset_root: str, batch_size: int, test: bool,
            object_as_task: bool):
        if not os.path.isdir(dataset_root):
            print(f"[!] {dataset_root} is not a valid directory!")
            exit(1)
        dataset = dataset.lower()
        if dataset == "obman":
            return ObManTaskLoader(dataset_root, batch_size, test=test, object_as_task=object_as_task)
        elif dataset == "fphad":
            return FPHADTaskLoader(dataset_root, batch_size, test=test, object_as_task=object_as_task)
        elif dataset == "ho3d":
            raise NotImplementedError("HO-3D Dataset Loader not implemented!")
        else:
            raise NotImplementedError(f"{dataset} Dataset Loader not implemented!")


class ObManTaskLoader(BaseDatasetTaskLoader):
    '''
    Refer to https://github.com/hassony2/obman/blob/master/obman/obman.py
    '''
    def __init__(self, root: str, batch_size: int, test: bool = False, object_as_task: bool = True):
        super().__init__(root, batch_size, test, object_as_task)
        # TODO:
        # 1. Load meta files, add index and meta info to the dictionary with the object as key
        # 2. Go through the dictionary and load the RGB image, the 2D and 3D labels
        # 3. Build a DataLoader?
        # 4. Use l2l's TaskDataSet module?
        self.samples = self._load(root, object_as_task=object_as_task)

    def _load_test(self, object_as_task: bool):
        indices = [int(x) for x.split('.')[0] in sorted(os.listdir(self._root))]
        print(indices)
        for idx in indices:
            pass


class FPHADTaskLoader(BaseDatasetTaskLoader):
    '''
    REMEMBER:
        "In addition to the samples in the FPHA dataset, we augment the 2D points with Gaussian noise
        (μ = 0, σ = 10) to help improve robustness to errors."
    '''
    def __init__(self, root: str, batch_size: int, test: bool = False, object_as_task: bool = True):
        super().__init__(root, batch_size, test, object_as_task)

    def _load_train_val(self, object_as_task: bool):
        trainset = Dataset(root=self._root, load_set='train', transform=self._transform)
        valset = Dataset(root=self._root, load_set='val', transform=self._transform)
        train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=self._batch_size,
                shuffle=True, num_workers=16)
        val_data_loader = torch.utils.data.DataLoader(valset, batch_size=self._batch_size,
                shuffle=False, num_workers=8)
        return train_data_loader, val_data_loader

