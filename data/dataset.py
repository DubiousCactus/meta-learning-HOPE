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

from data.custom import CustomDataset, CustomTaskDataset, CompatDataLoader
from HOPE.utils.dataset import Dataset


from typing import Tuple, Dict, List
from abc import abstractmethod, ABC
from tqdm import tqdm


class BaseDatasetTaskLoader(ABC):
    _transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    def __init__(
        self, root: str, batch_size: int, k_shots: int, test: bool, object_as_task: bool
    ):
        self._root = root
        self._batch_size = batch_size
        self.k_shots = k_shots
        self.train, self.val, self.test = None, None, None
        if test:
            self.test = self._load_test(object_as_task)
        else:
            self.train, self.val = self._load_train_val(object_as_task)

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
    ):
        super().__init__(root, batch_size, k_shots, test, object_as_task)
        self.num_objects = -1
        # TODO:
        # 1. Load meta files, add index and meta info to the dictionary with the object as key
        # 2. Go through the dictionary and load the RGB image, the 2D and 3D labels
        # 3. Build a DataLoader?
        # 4. Use l2l's TaskDataSet module?

    def _load_as_tasks(self, root) -> CustomTaskDataset:
        samples = {}
        class_ids = []
        indices = {
            x.split(".")[0]: os.path.join(root, "meta", x)
            for x in sorted(os.listdir(os.path.join(root, "meta")))
        }
        for idx, meta in tqdm(indices.items()):
            with open(meta, "rb") as meta_file:
                meta_obj = pickle.load(meta_file)
                obj_id = meta_obj["class_id"]
                img_path = os.path.join(root, "rgb", f"{idx}.jpg")
                # TODO:
                p_2d, p_3d = torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])
                if obj_id in class_ids:
                    samples[class_ids.index(obj_id)].append((img_path, p_2d, p_3d))
                else:
                    class_ids.append(obj_id)
                    samples[class_ids.index(obj_id)] = [(img_path, p_2d, p_3d)]
        self.num_objects = len(list(samples.keys()))
        print(f"[*] Loaded {self.num_objects} object classes from {root}!")
        return CustomTaskDataset(samples, self._transform)

    def _load(self, root) -> CustomDataset:
        samples = []
        indices = {
            x.split(".")[0]: os.path.join(root, "meta", x)
            for x in sorted(os.listdir(os.path.join(root, "meta")))
        }
        for idx, meta in tqdm(indices.items()):
            with open(meta, "rb") as meta_file:
                meta_obj = pickle.load(meta_file)
                img_path = os.path.join(root, "rgb", f"{idx}.jpg")
                # TODO:
                p_2d, p_3d = torch.zeros((29,2)), zeros((29,3))
                samples.append((img_path, p_2d, p_3d))
        return CustomDataset(samples, self._transform)

    def _load_train_val(self, object_as_task: bool) -> Tuple[dict, dict]:
        if "train" in os.listdir(self._root):
            train_path = os.path.join(self._root, "train")
            val_path = os.path.join(self._root, "val")
        else:
            raise Exception(
                f"{self._root} directory does not contain the 'train' folder!"
            )
        if object_as_task:
            # train_dataset = l2l.data.MetaDataset(
            # self._load_as_tasks(train_path), indices_to_labels=self._shapenet_labels
            # )
            val_task_set = self._load_as_tasks(val_path)
            val_dataset = l2l.data.MetaDataset(
                val_task_set, indices_to_labels=val_task_set.class_labels
            )
            # t_transforms = [
            # l2l.data.transforms.NWays(train_dataset, n=self.num_objects),
            # l2l.data.transforms.KShots(train_dataset, k=self.k_shots),
            # l2l.data.transforms.LoadData(train_dataset),
            # ]
            v_transforms = [
                l2l.data.transforms.NWays(val_dataset, n=self.num_objects),
                l2l.data.transforms.KShots(val_dataset, k=self.k_shots),
                l2l.data.transforms.LoadData(val_dataset),
            ]
            # train_dataset_loader = l2l.data.TaskDataset(train_dataset, t_transforms)
            val_dataset_loader = l2l.data.TaskDataset(val_dataset, v_transforms)
        else:
            # train_dataset_loader = torch.utils.data.DataLoader(
            # self._load(train_path), self._batch_size, shuffle=True, num_workers=8
            # )
            val_dataset_loader = CompatDataLoader(
                self._load(val_path), self._batch_size, shuffle=False, num_workers=4
            )
        return val_dataset_loader, val_dataset_loader

    def _load_test(self, object_as_task: bool) -> dict:
        if "test" in os.listdir(self._root):
            path = os.path.join(self._root, "test")
        else:
            raise Exception(
                f"{self._root} directory does not contain the 'test' folder!"
            )
        if object_as_task:
            test = self._load_as_tasks(path)
        else:
            test = {}
        return test


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
    ):
        super().__init__(root, batch_size, k_shots, test, object_as_task)

    def _load_train_val(self, object_as_task: bool) -> Tuple[dict, dict]:
        trainset = Dataset(root=self._root, load_set="train", transform=self._transform)
        valset = Dataset(root=self._root, load_set="val", transform=self._transform)
        train_data_loader = CompatDataLoader(
            trainset, batch_size=self._batch_size, shuffle=True, num_workers=16
        )
        val_data_loader = CompatDataLoader(
            valset, batch_size=self._batch_size, shuffle=False, num_workers=8
        )
        return train_data_loader, val_data_loader
