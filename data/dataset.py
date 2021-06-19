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
import learn2learn as l2l
import pickle
import torch
import os

from data.custom import CustomDataset, CompatDataLoader
from HOPE.utils.dataset import Dataset

from typing import Tuple, Dict, List, Union
from abc import abstractmethod, ABC
from tqdm import tqdm


class BaseDatasetTaskLoader(ABC):
    _transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        test: bool,
        object_as_task: bool,
        use_cuda: bool,
        gpu_number: int,
    ):
        self._root = root
        self._batch_size = batch_size
        self.k_shots = k_shots
        self._use_cuda = use_cuda
        self._gpu_number = gpu_number
        self.train, self.val, self.test = None, None, None
        if test:
            self.test = self._load(object_as_task, "test", False)
        else:
            self.train, self.val = self._load(
                object_as_task, "train", True
            ), self._load(object_as_task, "val", False)

    def _load_test(self, object_as_task: bool) -> dict:
        raise NotImplementedError

    def _load_train_val(self, object_as_task: bool) -> Tuple[dict, dict]:
        raise NotImplementedError


class ObManTaskLoader(BaseDatasetTaskLoader):
    """
    Refer to https://github.com/hassony2/obman/blob/master/obman/obman.py
    ObMan consists of 8 object categories from ShapeNet (bottles, bowls, cans, jars, knifes,
    cellphones, cameras and remote controls) for a total of 2772 meshes.
    """

    _shapenet_labels = {
        0: "jar",
        1: "can",
        2: "camera",
        3: "bottle",
        4: "cellphone",
        5: "bowl",
        6: "knife",
        7: "remote control",
    }

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots,
        test: bool = False,
        object_as_task: bool = True,
        use_cuda: bool = True,
        gpu_number: int = 0,
    ):
        super().__init__(
            root, batch_size, k_shots, test, object_as_task, use_cuda, gpu_number
        )

    def _make_dataset(self, root, object_as_task=False) -> CustomDataset:
        samples = {} if object_as_task else []
        class_ids = []
        indices = {
            x.split(".")[0]: os.path.join(root, "meta", x)
            for x in sorted(os.listdir(os.path.join(root, "meta")))
        }
        for idx, meta in tqdm(indices.items()):
            with open(meta, "rb") as meta_file:
                meta_obj = pickle.load(meta_file)
                img_path = os.path.join(root, "rgb", f"{idx}.jpg")
                # TODO:
                # print(meta_obj.keys())
                # print(meta_obj['pose'].shape, meta_obj['verts_3d'].shape, meta_obj['coords_2d'].shape, meta_obj['coords_3d'].shape)
                p_2d, p_3d = torch.zeros((29, 2)), torch.zeros((29, 3))
                if object_as_task:
                    obj_id = meta_obj["class_id"]
                    if obj_id in class_ids:
                        samples[class_ids.index(obj_id)].append((img_path, p_2d, p_3d))
                    else:
                        class_ids.append(obj_id)
                        samples[class_ids.index(obj_id)] = [(img_path, p_2d, p_3d)]
                else:
                    samples.append((img_path, p_2d, p_3d))
        return CustomDataset(
            samples,
            self._transform,
            object_as_task=object_as_task,
            use_cuda=self._use_cuda,
            gpu_number=self._gpu_number,
        )

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool
    ) -> Union[CompatDataLoader, l2l.data.TaskDataset]:
        if split in os.listdir(self._root):
            split_path = os.path.join(self._root, split)
        else:
            raise Exception(
                f"{self._root} directory does not contain the '{split}' folder!"
            )
        pickle_path = os.path.join(split_path, f"{split}_task_pickle.pkl" if object_as_task else f"{split}_pickle.pkl")
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading {split} split from {pickle_path}...")
                split_task_set = pickle.load(pickle_file)
        else:
            split_task_set = self._make_dataset(split_path, object_as_task=object_as_task)
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving {split} split into {pickle_path}...")
                pickle.dump(split_task_set, pickle_file)
        if object_as_task:
            split_dataset = l2l.data.MetaDataset(
                split_task_set, indices_to_labels=split_task_set.class_labels
            )
            split_dataset_loader = l2l.data.TaskDataset(
                split_dataset,
                [
                    l2l.data.transforms.NWays(split_dataset, n=1),
                    l2l.data.transforms.KShots(split_dataset, k=self.k_shots),
                    l2l.data.transforms.LoadData(split_dataset),
                ],
            )
        else:
            split_dataset_loader = CompatDataLoader(
                split_task_set,
                self._batch_size,
                shuffle=shuffle,
                num_workers=8,
                use_cuda=self._use_cuda,
                gpu_number=self._gpu_number,
            )
        return split_dataset_loader


class FPHADTaskLoader(BaseDatasetTaskLoader):
    """
    REMEMBER:
        "In addition to the samples in the FPHA dataset, we augment the 2D points with Gaussian noise
        (μ = 0, σ = 10) to help improve robustness to errors."
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        test: bool = False,
        object_as_task: bool = True,
        use_cuda: bool = True,
        gpu_number: int = 0,
    ):
        super().__init__(
            root, batch_size, k_shots, test, object_as_task, use_cuda, gpu_number
        )

    # TODO: TaskLoader
    def _load_train_val(self, object_as_task: bool) -> Tuple[dict, dict]:
        trainset = Dataset(root=self._root, load_set="train", transform=self._transform)
        valset = Dataset(root=self._root, load_set="val", transform=self._transform)
        train_data_loader = CompatDataLoader(
            trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=16,
            use_cuda=self._use_cuda,
            gpu_number=self._gpu_number,
        )
        val_data_loader = CompatDataLoader(
            valset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=8,
            use_cuda=self._use_cuda,
            gpu_number=self._gpu_number,
        )
        return train_data_loader, val_data_loader
