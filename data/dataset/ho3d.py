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
from typing import Union, Tuple, List
from functools import reduce
from copy import copy

import learn2learn as l2l
import numpy as np
import itertools
import trimesh
import pickle
import torch
import os


def kp2d_transform(keypoints):
    _min, _max = -319.3636, 1541.3502
    return (keypoints - _min) / (_max - _min)


class HO3DTaskLoader(BaseDatasetTaskLoader):
    """
    The official spltis won't be used for this dataset. Custom splits will be used, with a
    specified hold-out value for how many object categories to hold out of the training set.

    Object categories:
    1: Banana
    2: Potted meat can
    3: Mustard bottl
    4: Sugar box
    5: Power drill
    6: Scissors
    7: Bleach cleanser
    8: Mug
    9: Cracker box
    10: Pitcher base <- in the evaluation set, no joint poses for the hand except the wrist
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        n_queries: int,
        test: bool = False,
        object_as_task: bool = True,
        hold_out: int = 0,
        normalize_keypoints: bool = False,
        use_cuda: bool = True,
        gpu_number: int = 0,
    ):
        super().__init__(
            root,
            batch_size,
            k_shots,
            n_queries,
            test,
            object_as_task,
            normalize_keypoints,
            use_cuda,
            gpu_number,
            auto_load=False,
        )
        # Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/ho/make_data.py.
        self._reorder_idx = np.array(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        )
        self._cam_extr = np.array(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )
        self._obj_labels = [
            "011_banana",
            "010_potted_meat_can",
            "006_mustard_bottle",
            "004_sugar_box",
            "035_power_drill",
            "037_scissors",
            "021_bleach_cleanser",
            "025_mug",
            "003_cracker_box",
            # "019_pitcher_base",
        ]
        self._split_categories = self._make_split_categories(hold_out, manual=True)
        print(f"[*] Training with {', '.join([self._obj_labels[i] for i in self._split_categories['train']])}")
        print(f"[*] Validating with {', '.join([self._obj_labels[i] for i in self._split_categories['val']])}")
        print(f"[*] Testing with {', '.join([self._obj_labels[i] for i in self._split_categories['test']])}")
        # Don't auto load, this is a custom loading
        if test:
            self.test = self._load(
                object_as_task,
                "test",
                ["train"],
                False,
                normalize_keypoints,
            )
        else:
            self.train, self.val = self._load(
                object_as_task,
                "train",
                ["train"],
                True,
                normalize_keypoints,
            ), self._load(
                object_as_task,
                "val",
                ["train"],
                False,
                normalize_keypoints,
            )

    def _make_split_categories(self, hold_out, manual=False) -> dict:
        """
        HO-3D contains 9 object categories. This method distributes those categories per split,
        according to "hold_out" which corresponds to how many are held out of the train split.
        However if N categories are held out, 2*N categories must be effectively held out because
        they can't be the same for the validation and test splits!
        """
        categories = list(range(len(self._obj_labels)))
        assert (
            len(self._obj_labels) - (2 * hold_out) >= 1
        ), "There must remain at least one category in the train split"
        if manual and hold_out == 2:
            splits = {
                "train": [1, 2, 3, 6, 8],
                "val": [0, 7], # Banana and Mug
                "test": [4, 5] # Power drill and Scissors
            }
        else:
            splits = {
                "train": categories[: -2 * hold_out],
                "val": categories[-2 * hold_out : -hold_out],
                "test": categories[-hold_out:],
            }
        return splits

    def _compute_labels(
        self, root: str, meta: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cam_intr = meta["camMat"]
        if meta["handJoints3D"] is None:
            raise ValueError
        if root != "evaluation":
            hand3d = meta["handJoints3D"][self._reorder_idx]
        else:
            # The official "evaluation" split only contains the wrist annotations
            hand3d = np.repeat(np.expand_dims(meta["handJoints3D"], 0), 21, 0)
            hand3d = hand3d[self._reorder_idx]
        obj_corners = meta["objCorners3D"]
        hand_obj_3d = np.concatenate([hand3d, obj_corners])
        hand_obj_3d = hand_obj_3d.dot(self._cam_extr.T)
        hand_obj_2d = cam_intr.dot(hand_obj_3d.transpose()).transpose()
        hand_obj_2d = (hand_obj_2d / hand_obj_2d[:, 2:])[:, :2]
        return torch.Tensor(hand_obj_2d), torch.Tensor(hand_obj_3d)

    def _make_dataset(
        self,
        split: str,
        dataset_root: str,
        root_folders: List[str],
        object_as_task=False,
        normalize_keypoints=False,
    ) -> CustomDataset:
        """
        Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/ho/make_data.py.
        """
        pickle_path = os.path.join(dataset_root, f"ho3d.pkl")
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading dataset from {pickle_path}...")
                samples = pickle.load(pickle_file)
        else:
            print(f"[*] Building dataset...")
            samples = {}
            failed = 0
            for rel_root in root_folders:
                if rel_root in os.listdir(self._root):
                    root = os.path.join(self._root, rel_root)
                else:
                    raise Exception(
                        f"{self._root} directory does not contain the '{rel_root}' folder!"
                    )
                for subject in os.listdir(root):
                    s_path = os.path.join(root, subject)
                    if not os.path.isdir(s_path):
                        continue
                    meta_dir = os.path.join(s_path, "meta")
                    for img in os.listdir(os.path.join(s_path, "rgb")):
                        file_no = img.split(".")[0]
                        img_path = os.path.join(s_path, "rgb", img)
                        meta_file = os.path.join(meta_dir, f"{file_no}.pkl")
                        meta = np.load(meta_file, allow_pickle=True)
                        try:
                            points_2d, points_3d = self._compute_labels(rel_root, meta)
                            # Rescale the 2D keypoints, because the images are rescaled from 640x480 to
                            # 224x224! This improves the performance of the 2D KP estimation GREATLY.
                            points_2d[:, 0] = points_2d[:, 0] * 224.0 / 640.0
                            points_2d[:, 1] = points_2d[:, 1] * 224.0 / 480.0
                        except ValueError:
                            failed += 1
                            continue
                        obj_class_id = self._obj_labels.index(meta["objName"])
                        if obj_class_id not in samples.keys():
                            samples[obj_class_id] = []
                        samples[obj_class_id].append((img_path, points_2d, points_3d))
            if failed != 0:
                print(f"[!] {failed} samples were missing annotations!")
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving {split} split into {pickle_path}...")
                pickle.dump(samples, pickle_file)
        if normalize_keypoints:
            print(f"[*] Normalizing 2D and 3D keypoints...")
            flat_samples = [s for sublist in samples.values() for s in sublist]
            kp_2d = torch.flatten(torch.vstack([p2d for _, p2d, _ in flat_samples]))
            min_2d, max_2d = (
                torch.min(kp_2d),
                torch.max(kp_2d),
            )
            print(min_2d, max_2d)
        # Hold out
        keys = samples.copy().keys()
        for category_id, _ in enumerate(keys):
            if category_id not in self._split_categories[split]:
                del samples[list(keys)[category_id]]
        print(
            f"[*] Loaded {reduce(lambda x, y: x + y, [len(x) for x in samples.values()])} samples from the {split} split."
        )
        print(f"[*] Total object categories: {len(samples.keys())}")
        if not object_as_task:  # Transform to list
            samples = list(itertools.chain.from_iterable(samples.values()))
        print(f"[*] Generating dataset in pinned memory...")
        dataset = CustomDataset(
            samples,
            img_transform=self._img_transform,
            kp2d_transform=kp2d_transform if normalize_keypoints else None,
            object_as_task=object_as_task,
        )

        return dataset

    def _load(
        self,
        object_as_task: bool,
        split: str,
        split_folders: List[str],
        shuffle: bool,
        normalize_keypoints: bool,
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        dataset = self._make_dataset(
            split,
            self._root,
            split_folders,
            object_as_task=object_as_task,
            normalize_keypoints=normalize_keypoints,
        )
        if object_as_task:
            split_dataset = l2l.data.MetaDataset(
                dataset, indices_to_labels=dataset.class_labels
            )
            split_dataset_loader = l2l.data.TaskDataset(
                split_dataset,
                [
                    l2l.data.transforms.NWays(split_dataset, n=1),
                    l2l.data.transforms.KShots(
                        split_dataset, k=self.k_shots + self.n_queries
                    ),
                    l2l.data.transforms.LoadData(split_dataset),
                ],
                num_tasks=(-1 if split == "train" else (len(dataset) / (self.k_shots + self.n_queries))),
            )
        else:
            split_dataset_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=8,
            )
        return split_dataset_loader
