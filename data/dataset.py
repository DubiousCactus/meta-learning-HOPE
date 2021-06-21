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
import numpy as np
import trimesh
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

    def _make_dataset(self, root, object_as_task=False) -> CustomDataset:
        raise NotImplementedError

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool
    ) -> Union[CompatDataLoader, l2l.data.TaskDataset]:
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
                hand_coords_2d, hand_coords_3d = torch.Tensor(
                    meta_obj["coords_2d"]
                ), torch.Tensor(meta_obj["coords_3d"])
                # TODO:
                obj_coords_2d, obj_coords_3d = torch.zeros((8, 2)), torch.zeros((8, 3))
                p_2d, p_3d = torch.cat([hand_coords_2d, obj_coords_2d]), torch.cat(
                    [hand_coords_3d, obj_coords_3d]
                )
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
        # Pickling doesn't improve the speed, might remove:
        # pickle_path = os.path.join(split_path, f"{split}_task_pickle.pkl" if object_as_task else f"{split}_pickle.pkl")
        # if os.path.isfile(pickle_path):
        # with open(pickle_path, "rb") as pickle_file:
        # print(f"[*] Loading {split} split from {pickle_path}...")
        #   split_task_set = pickle.load(pickle_file)
        # else:
        # split_task_set = self._make_dataset(split_path, object_as_task=object_as_task)
        # with open(pickle_path, "wb") as pickle_file:
        # print(f"[*] Saving {split} split into {pickle_path}...")
        #     pickle.dump(split_task_set, pickle_file)
        split_task_set = self._make_dataset(split_path, object_as_task=object_as_task)
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
    # TODO:
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

    # Loading utilities
    def _load_objects(self, root):
        object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            all_models[obj_name] = mesh
        return all_models


    def _get_skeleton(self, sample, root):
        skeleton_path = os.path.join(root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[sample['frame_idx']]
        return skeleton


    def _get_obj_transform(self, sample, root):
        seq_path = os.path.join(root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        #print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def _make_dataset(self, split: str, object_as_task=False) -> CustomDataset:
        '''
        Refer to the make_datay.py script in the HOPE project: ../HOPE/datasets/fhad/make_data.py.
        '''
        skeleton_root = os.path.join(self._root, 'Hand_pose_annotation_v1')
        obj_root = os.path.join(self._root, 'Object_models')
        obj_trans_root = os.path.join(self._root, 'Object_6D_pose_annotation_v1_1')
        file_root = os.path.join(self._root, 'Video_files')
        # Load object mesh
        object_infos = self._load_objects(obj_root)
        reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
        seq_splits = {'train': None, 'val': 1, 'test': 3}

        cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
            [0, 0, 0, 1]])
        cam_intr = np.array([[1395.749023, 0, 935.732544],
            [0, 1395.749268, 540.681030],
            [0, 0, 1]])

        samples = []

        for root, dirs, files in os.walk(obj_trans_root):
            if 'object_pose.txt' in files:
                path = root.split(os.sep)
                subject, action_name, seq_idx = path[-3], path[-2], path[-1]
                if seq_idx != seq_splits[split] and seq_splits[split] is not None:
                    continue
                obj = '_'.join(action_name.split('_')[1:])
                video_seq = os.path.join(file_root, subject, action_name, seq_idx, 'color')
                if not os.path.isdir(video_seq):
                    # print(f"[!] {video_seq} is missing!")
                    continue
                for file_name in os.listdir(video_seq):
                    frame_idx = int(file_name.split('.')[0].split('_')[1])
                    sample = {
                        'subject': subject,
                        'action_name': action_name,
                        'seq_idx': seq_idx,
                        'frame_idx': frame_idx,
                        'object': obj
                    }
                    img_path = os.path.join(video_seq, file_name)
                    skel = self._get_skeleton(sample, skeleton_root)[reorder_idx]
                    # Load object transform
                    obj_trans = self._get_obj_transform(sample, obj_trans_root)
                    mesh = object_infos[sample['object']]
                    verts = np.array(mesh.bounding_box_oriented.vertices) * 1000
                    # Apply transform to object
                    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
                    verts_trans = obj_trans.dot(hom_verts.T).T
                    # Apply camera extrinsic to object
                    verts_camcoords = cam_extr.dot(verts_trans.transpose()).transpose()[:, :3]
                    # Project and object skeleton using camera intrinsics
                    verts_hom2d = np.array(cam_intr).dot(verts_camcoords.transpose()).transpose()
                    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
                    # Apply camera extrinsic to hand skeleton
                    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
                    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]
                    points = np.concatenate((skel_camcoords, verts_camcoords))
                    projected_points = np.concatenate((skel_proj, verts_proj))
                    samples.append((img_path, projected_points, points))

        return CustomDataset(
                samples,
                self._transform,
                object_as_task=False,
                use_cuda=self._use_cuda,
                gpu_number=self._gpu_number,
            )

    def _load(
        self, object_as_task: bool, split: str, shuffle: bool
    ) -> Union[CompatDataLoader, l2l.data.TaskDataset]:
        if object_as_task:
            split_dataset = self._make_dataset('train', object_as_task=True)
            split_data_loader = CompatDataLoader(
                split_dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=8,
                use_cuda=self._use_cuda,
                gpu_number=self._gpu_number,
            )
        else:
            split_dataset = Dataset(
                root=self._root, load_set=split, transform=self._transform
            )
            split_data_loader = CompatDataLoader(
                split_dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=8,
                use_cuda=self._use_cuda,
                gpu_number=self._gpu_number,
            )
        return split_data_loader
