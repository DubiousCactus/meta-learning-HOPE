#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
ObMan Dataset (task) loader
"""

from data.dataset.base import BaseDatasetTaskLoader
from util.utils import compute_OBB_corners
from torch.utils.data import DataLoader
from data.custom import CustomDataset
from util.utils import fast_load_obj
from functools import reduce
from typing import Union
from tqdm import tqdm

import learn2learn as l2l
import numpy as np
import trimesh
import pickle
import torch
import os


def kp2d_transform(keypoints):
    _min, _max = -234.1035, 482.4938
    return (keypoints - _min) / (_max - _min)


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
        k_shots: int,
        n_querries: int,
        shapenet_root: str,
        test: bool = False,
        object_as_task: bool = True,
        normalize_keypoints: bool = False,
        hand_only: bool = True,
        use_cuda: bool = True,
        gpu_number: int = 0,
    ):
        # Taken from https://github.com/hassony2/obman
        if shapenet_root[-1] == "/":
            shapenet_root = shapenet_root[:-1]
            if not os.path.isdir(shapenet_root):
                raise Exception(f"ShapeNet root not found: {shapenet_root}")
        self._shapenet_template = shapenet_root + "/{}/{}/models/model_normalized.pkl"
        self._cam_intr = np.array(
            [[480.0, 0.0, 128.0], [0.0, 480.0, 128.0], [0.0, 0.0, 1.0]]
        ).astype(np.float32)

        self._cam_extr = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]
        ).astype(np.float32)
        self._bboxes = {}
        super().__init__(
            root,
            batch_size,
            k_shots,
            n_querries,
            test,
            object_as_task,
            normalize_keypoints,
            hand_only,
            use_cuda,
            gpu_number,
        )

    def _load_mesh(self, model_path: str) -> trimesh.Trimesh:
        """
        Directly copied from: https://github.com/hassony2/obman
        """
        model_path_obj = model_path.replace(".pkl", ".obj")
        if os.path.exists(model_path):
            with open(model_path, "rb") as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            with open(model_path_obj, "r") as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                "Could not find model pkl or obj file at {}".format(
                    model_path.split(".")[-2]
                )
            )
        return trimesh.load(mesh)

    def _compute_labels(self, meta_info: dict) -> tuple:
        # Get the hand coordinates
        hand_coords_2d, hand_coords_3d = (
            torch.Tensor(meta_info["coords_2d"].astype(np.float32)),
            torch.Tensor(
                self._cam_extr[:3, :3]
                .dot(meta_info["coords_3d"].transpose())
                .transpose()
            ),
        )
        # 1. Load the mesh (see obman.py)
        obj_path = self._shapenet_template.format(
            meta_info["class_id"], meta_info["sample_id"]
        )
        mesh = self._load_mesh(obj_path)
        # 2. Load the transform
        transform = meta_info["affine_transform"]
        # 3. Obtain the oriented bounding box vertices (x1000?)
        if hash(mesh) not in self._bboxes:
            verts = compute_OBB_corners(mesh)  # * 1000
            self._bboxes[hash(mesh)] = verts
        else:
            verts = self._bboxes[hash(mesh)]
        # 4. Apply the transform to the vertices
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
        trans_verts = transform.dot(hom_verts.T).T[:, :3]
        # 5. Apply the camera extrinsic to the transformed vertices: these are the 3D vertices
        trans_verts = self._cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
        vertices_3d = np.array(trans_verts).astype(np.float32)
        # 6. Project using camera intrinsics: these are the 2D vertices
        hom_2d_verts = np.dot(self._cam_intr, vertices_3d.transpose())
        vertices_2d = hom_2d_verts / hom_2d_verts[2, :]
        vertices_2d = vertices_2d[:2, :].transpose()
        return (
            torch.cat([hand_coords_2d, torch.Tensor(vertices_2d)]),
            torch.cat([hand_coords_3d, torch.Tensor(vertices_3d)]),
        )

    def _make_dataset(
        self, split: str, root: str, object_as_task=False, normalize_keypoints=False
    ) -> CustomDataset:
        pickle_path = os.path.join(
            root,
            f"obman_{split}_task.pkl" if object_as_task else f"obman_{split}.pkl",
        )
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading {split} split from {pickle_path}...")
                samples = pickle.load(pickle_file)
        else:
            print(f"[*] Building {split} split...")
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
                    coord_2d, coord_3d = self._compute_labels(meta_obj)
                    # Rescale the 2D keypoints, because the images are rescaled from 256x256 to
                    # 224x224! This improves the performance of the 2D KP estimation GREATLY.
                    coord_2d[:, 0] = coord_2d[:, 0] * 224.0 / 256.0
                    coord_2d[:, 1] = coord_2d[:, 1] * 224.0 / 256.0
                    if object_as_task:
                        obj_id = meta_obj["class_id"]
                        if obj_id in class_ids:
                            samples[class_ids.index(obj_id)].append(
                                (img_path, coord_2d, coord_3d)
                            )
                        else:
                            class_ids.append(obj_id)
                            samples[class_ids.index(obj_id)] = [
                                (img_path, coord_2d, coord_3d)
                            ]
                    else:
                        samples.append((img_path, coord_2d, coord_3d))
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving {split} split into {pickle_path}...")
                pickle.dump(samples, pickle_file)
        if normalize_keypoints:
            print(f"[*] Normalizing 2D and 3D keypoints...")
            if object_as_task:
                flat_samples = [s for sublist in samples.values() for s in sublist]
                kp_2d = torch.flatten(torch.vstack([p2d for _, p2d, _ in flat_samples]))
                min_2d, max_2d = (
                    torch.min(kp_2d),
                    torch.max(kp_2d),
                )
                print(min_2d, max_2d)

        if object_as_task:
            print(
                f"[*] Loaded {reduce(lambda x, y: x + y, [len(x) for x in samples.values()])} samples from the {split} split."
            )
            print(f"[*] Total object categories: {len(samples.keys())}")
        else:
            print(f"[*] Loaded {len(samples)} samples from the {split} split.")
        print(f"[*] Generating dataset in pinned memory...")
        dataset = CustomDataset(
            samples,
            img_transform=self._img_transform,
            object_as_task=object_as_task,
            hand_only=self._hand_only,
            kp2d_transform=kp2d_transform if normalize_keypoints else None,
        )
        return dataset

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool, normalize_keypoints: bool
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        if split in os.listdir(self._root):
            split_path = os.path.join(self._root, split)
        else:
            raise Exception(
                f"{self._root} directory does not contain the '{split}' folder!"
            )
        split_task_set = self._make_dataset(
            split,
            split_path,
            object_as_task=object_as_task,
            normalize_keypoints=normalize_keypoints,
        )
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
                num_workers=min(os.cpu_count() - 1, 8),
            )
        return split_dataset_loader
