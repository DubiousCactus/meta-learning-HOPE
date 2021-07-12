#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
First-Person Hand Dataset (task) loader
"""

from data.dataset.base import BaseDatasetTaskLoader
from torch.utils.data import DataLoader
from data.custom import CustomDataset
from functools import reduce
from typing import Union

import learn2learn as l2l
import numpy as np
import trimesh
import pickle
import torch
import os


def kp2d_transform(keypoints):
    """
    "In addition to the samples in the FPHA dataset, we augment the 2D points with Gaussian noise
    (μ = 0, σ = 10) to help improve robustness to errors."
    """
    std = 10
    return keypoints + (std * torch.randn(keypoints.shape))


class FPHADTaskLoader(BaseDatasetTaskLoader):

    _object_class = ["juice_bottle", "liquid_soap", "milk", "salt"]

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
        augment_2d: bool = False,
    ):
        # Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/fhad/make_data.py.
        self._object_infos = self._load_objects(os.path.join(root, "Object_models"))
        self._obj_trans_root = os.path.join(root, "Object_6D_pose_annotation_v1_1")
        self._file_root = os.path.join(root, "Video_files")
        self._seq_splits = {"train": None, "val": 1, "test": 3}
        self._skeleton_root = os.path.join(root, "Hand_pose_annotation_v1")
        self._augment_2d = augment_2d
        self._reorder_idx = np.array(
            [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
        )
        self._cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self._cam_intr = np.array(
            [[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]]
        )
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

    def _load_objects(self, root):
        all_models = {}
        for obj_name in self._object_class:
            obj_path = os.path.join(
                root, "{}_model".format(obj_name), "{}_model.ply".format(obj_name)
            )
            mesh = trimesh.load(obj_path)
            all_models[obj_name] = mesh
        return all_models

    def _get_skeleton(self, sample, root):
        skeleton_path = os.path.join(
            root,
            sample["subject"],
            sample["action_name"],
            sample["seq_idx"],
            "skeleton.txt",
        )
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[
            sample["frame_idx"]
        ]
        return skeleton

    def _get_obj_transform(self, sample, root):
        seq_path = os.path.join(
            root,
            sample["subject"],
            sample["action_name"],
            sample["seq_idx"],
            "object_pose.txt",
        )
        with open(seq_path, "r") as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample["frame_idx"]]
        line = raw_line.strip().split(" ")
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        return trans_matrix

    def _compute_labels(
        self,
        file_name: str,
        obj_name: str,
        subject: str,
        action_name: str,
        seq_idx: str,
    ) -> tuple:
        frame_idx = int(file_name.split(".")[0].split("_")[1])
        sample = {
            "subject": subject,
            "action_name": action_name,
            "seq_idx": seq_idx,
            "frame_idx": frame_idx,
            "object": obj_name,
        }
        skel = self._get_skeleton(sample, self._skeleton_root)[self._reorder_idx]
        # Load object transform
        obj_trans = self._get_obj_transform(sample, self._obj_trans_root)
        mesh = self._object_infos[obj_name]
        verts = np.array(mesh.bounding_box_oriented.vertices) * 1000
        # Apply transform to object
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
        verts_trans = obj_trans.dot(hom_verts.T).T
        # Apply camera extrinsic to object
        verts_camcoords = self._cam_extr.dot(verts_trans.transpose()).transpose()[:, :3]
        # Project and object skeleton using camera intrinsics
        verts_hom2d = (
            np.array(self._cam_intr).dot(verts_camcoords.transpose()).transpose()
        )
        verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
        # Apply camera extrinsic to hand skeleton
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = (
            self._cam_extr.dot(skel_hom.transpose())
            .transpose()[:, :3]
            .astype(np.float32)
        )
        skel_hom2d = (
            np.array(self._cam_intr).dot(skel_camcoords.transpose()).transpose()
        )
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]
        return (
            torch.cat([torch.Tensor(skel_proj), torch.Tensor(verts_proj)]),
            torch.cat([torch.Tensor(skel_camcoords), torch.Tensor(verts_camcoords)]),
        )

    def _make_dataset(self, split: str, object_as_task=False) -> CustomDataset:
        """
        Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/fhad/make_data.py.
        """
        pickle_path = os.path.join(
            self._root,
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
            for root, _, files in os.walk(self._obj_trans_root):
                if "object_pose.txt" in files:
                    path = root.split(os.sep)
                    subject, action_name, seq_idx = path[-3], path[-2], path[-1]
                    if (
                        self._seq_splits[split] is not None
                        and int(seq_idx) != self._seq_splits[split]
                    ):
                        continue
                    obj_name = "_".join(action_name.split("_")[1:])
                    video_seq = os.path.join(
                        self._file_root, subject, action_name, seq_idx, "color"
                    )
                    if not os.path.isdir(video_seq):
                        # print(f"[!] {video_seq} is missing!")
                        continue
                    for file_name in os.listdir(video_seq):
                        img_path = os.path.join(video_seq, file_name)
                        points_2d, points_3d = self._compute_labels(
                            file_name, obj_name, subject, action_name, seq_idx
                        )
                        if object_as_task:
                            obj_class_id = self._object_class.index(obj_name)
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
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving {split} split into {pickle_path}...")
                pickle.dump(samples, pickle_file)
        print(f"[*] Generating dataset in pinned memory...")
        dataset = CustomDataset(
            samples,
            img_transform=self._img_transform,
            kp2d_transform=kp2d_transform
            if split == "train" and self._augment_2d
            else None,
            object_as_task=object_as_task,
        )
        return dataset

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        split_task_set = self._make_dataset(split, object_as_task=object_as_task)
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
