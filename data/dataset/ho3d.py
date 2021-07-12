#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Hand-Object 3D dataset (task) loader
"""

from data.dataset.base import BaseDatasetTaskLoader
from torch.utils.data import DataLoader
from data.custom import CustomDataset
from typing import Union, Tuple
from functools import reduce

import learn2learn as l2l
import numpy as np
import trimesh
import pickle
import torch
import os


class HO3DTaskLoader(BaseDatasetTaskLoader):

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        n_querries: int,
        test: bool = False,
        object_as_task: bool = True,
        use_cuda: bool = True,
        gpu_number: int = 0,
    ):
        # Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/ho/make_data.py.
        self._reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        self._cam_extr = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        self._seq_splits = {'train': None, 'val': 'MC6'}
        # Only call super() last, because the base class's init() calls the _load() function!
        super().__init__(
            root,
            batch_size,
            k_shots,
            n_querries,
            test,
            object_as_task,
            use_cuda,
            gpu_number,
        )


    def _compute_labels(self, split: str, meta: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        cam_intr = meta['camMat']
        if meta['handJoints3D'] is None:
            raise ValueError
        if split != "test":
            hand3d = meta['handJoints3D'][self._reorder_idx]
        else:
            hand3d = np.repeat(np.expand_dims(meta['handJoints3D'], 0), 21, 0)
            hand3d = hand3d[self._reorder_idx]
        obj_corners = meta['objCorners3D']
        hand_obj_3d = np.concatenate([hand3d, obj_corners])
        hand_obj_3d = hand_obj_3d.dot(self._cam_extr.T)
        hand_obj_2d = cam_intr.dot(hand_obj_3d.transpose()).transpose()
        hand_obj_2d = (hand_obj_2d / hand_obj_2d[:, 2:])[:, :2]
        return torch.Tensor(hand_obj_2d), torch.Tensor(hand_obj_3d)

    def _make_dataset(self, split: str, root: str, object_as_task=False) -> CustomDataset:
        """
        Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/ho/make_data.py.
        """
        pickle_path = os.path.join(
            root,
            f"fphad_{split}_task.pkl" if object_as_task else f"fphad_{split}.pkl",
        )
        samples = []
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading {split} split from {pickle_path}...")
                samples = pickle.load(pickle_file)
        else:
            print(f"[*] Building {split} split...")
            samples = {} if object_as_task else []
            failed = 0
            for subject in os.listdir(root):
                if (
                    self._seq_splits[split] is not None
                    and subject != self._seq_splits[split]
                ):
                    continue
                s_path = os.path.join(root, subject)
                meta_dir = os.path.join(s_path, 'meta')
                for img_path in os.listdir(os.path.join(s_path, 'rgb')):
                    file_no = img_path.split('.')[0]
                    meta_file = os.path.join(meta_dir, f"{file_no}.pkl")
                    meta = np.load(meta_file, allow_pickle=True)
                    try:
                        points_2d, points_3d = self._compute_labels(split, meta)
                    except ValueError:
                        failed += 1
                    obj_class_id = meta['objLabel']
                    if object_as_task:
                        if obj_class_id not in samples.keys():
                            samples[obj_class_id] = []
                        samples[obj_class_id].append(
                            (img_path, points_2d, points_3d)
                        )
                    else:
                        samples.append((img_path, points_2d, points_3d))
            if object_as_task:
                print(
                    f"[*] Loaded {reduce(lambda x, y: x + y, [len(x) for x in samples.values()])} samples from the {split} split."
                )
            else:
                print(f"[*] Loaded {len(samples)} samples from the {split} split.")
            if failed != 0:
                print(f"[!] {failed} samples were missing annotations!")
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving {split} split into {pickle_path}...")
                pickle.dump(samples, pickle_file)
        print(f"[*] Generating dataset in pinned memory...")
        dataset = CustomDataset(
            samples,
            img_transform=self._img_transform,
            object_as_task=object_as_task,
        )
        return dataset

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        if split in os.listdir(self._root):
            split_path = os.path.join(self._root, split)
        else:
            raise Exception(
                f"{self._root} directory does not contain the '{split}' folder!"
            )
        split_task_set = self._make_dataset(split, split_path, object_as_task=object_as_task)
        if object_as_task:
            split_dataset = l2l.data.MetaDataset(
                split_task_set, indices_to_labels=split_task_set.class_labels
            )
            split_dataset_loader = l2l.data.TaskDataset(
                split_dataset,
                [
                    l2l.data.transforms.NWays(split_dataset, n=1),
                    l2l.data.transforms.KShots(
                        split_dataset, k=self.k_shots + self.n_querries
                    ),
                    l2l.data.transforms.LoadData(split_dataset),
                ],
            )
        else:
            split_dataset_loader = DataLoader(
                split_task_set,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=8,
            )
        return split_dataset_loader
