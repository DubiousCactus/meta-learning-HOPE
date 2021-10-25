# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

from data.dataset.base import BaseDatasetTaskLoader
from torch.utils.data import DataLoader
from data.custom import CustomDataset
from util.utils import fast_load_obj
from typing import Union, Tuple
from functools import reduce

import learn2learn as l2l
import numpy as np
import itertools
import trimesh
import pickle
import torch
import yaml
import os


class DexYCBDatasetTaskLoader(BaseDatasetTaskLoader):
    _subjects = [
        "20200709-subject-01",
        # "20200813-subject-02",
        # "20200820-subject-03",
        # "20200903-subject-04",
        # "20200908-subject-05",
        # "20200918-subject-06",
        # "20200928-subject-07",
        # "20201002-subject-08",
        # "20201015-subject-09",
        # "20201022-subject-10",
    ]

    _serials = [
        "836212060125",
        "839512060362",
        "840412060917",
        "841412060263",
        "932122060857",
        "932122060861",
        "932122061900",
        "932122062010",
    ]

    _obj_labels = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick",
    ]

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
        # TODO: Reorder the joints as in the other datasets? What is it for? For the graph nets to
        # better model the joint neighbourings?
        self._calib_dir = os.path.join(self._root, "calibration")
        self._model_dir = os.path.join(self._root, "models")
        self._h = 480
        self._w = 640
        self._bboxes = {}  # Cache

        self._split_categories = self._make_split_categories(hold_out, manual=True)
        print(
            f"[*] Training with {', '.join([self._obj_labels[i] for i in self._split_categories['train']])}"
        )
        print(
            f"[*] Validating with {', '.join([self._obj_labels[i] for i in self._split_categories['val']])}"
        )
        print(
            f"[*] Testing with {', '.join([self._obj_labels[i] for i in self._split_categories['test']])}"
        )
        # Don't auto load, this is a custom loading
        if test:
            self.test = self._load(
                object_as_task,
                "test",
                False,
                normalize_keypoints,
            )
        else:
            self.train, self.val = self._load(
                object_as_task,
                "train",
                True,
                normalize_keypoints,
            ), self._load(
                object_as_task,
                "val",
                False,
                normalize_keypoints,
            )

    def _make_split_categories(self, hold_out, manual=False, seed=0) -> dict:
        """
        HO-3D contains 20 object categories. This method distributes those categories per split,
        according to "hold_out" which corresponds to how many are held out of the train split.
        However if N categories are held out, 2*N categories must be effectively held out because
        they can't be the same for the validation and test splits!
        """
        categories = list(range(len(self._obj_labels)))
        assert (
            len(self._obj_labels) - (2 * hold_out) >= 1
        ), "There must remain at least one category in the train split"
        # TODO: Random selection
        splits = {
            "train": categories[: -2 * hold_out],
            "val": categories[-2 * hold_out : -hold_out],
            "test": categories[-hold_out:],
        }
        return splits

    def _compute_labels(
            self, cam_intr: np.ndarray, meta: dict, labels: dict, obj_label: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The target object seems to be consistently the first
        # one. The format is [R; t] with R 3x3 and t 3x1.
        # Cache the mesh bounding box
        obj_file_path = os.path.join(
            self._root,
            "models",
            obj_label,
            "textured_simple.obj",
        )
        if obj_file_path not in self._bboxes:
            with open(obj_file_path, "r") as m_f:
                mesh = fast_load_obj(m_f)[0]
            mesh = trimesh.load(mesh)
            vert3d = np.array(mesh.bounding_box_oriented.vertices)
            self._bboxes[obj_file_path] = vert3d
        else:
            vert3d = self._bboxes[obj_file_path]
        # Apply the rotation + translation to the bbox vertices
        transform = labels["pose_y"][0]
        hom_verts = np.concatenate(
            [vert3d, np.ones([vert3d.shape[0], 1])],
            axis=1,
        )
        vert3d = transform.dot(hom_verts.T).T[:, :3]
        # Project to 2D
        hom_2d_verts = np.dot(cam_intr, vert3d.transpose())
        vert2d = hom_2d_verts / hom_2d_verts[2, :]
        vert2d = vert2d[:2, :].transpose()
        return (
            torch.cat([torch.Tensor(labels["joint_2d"][0]),torch.Tensor(vert2d)]),
            torch.cat([torch.Tensor(labels["joint_3d"][0]), torch.Tensor(vert3d)])
        )

    def _make_dataset(
        self,
        split: str,
        dataset_root: str,
        object_as_task=False,
        normalize_keypoints=False,
    ) -> CustomDataset:
        pickle_path = os.path.join(dataset_root, f"dexycb.pkl")
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading dataset from {pickle_path}...")
                samples = pickle.load(pickle_file)
        else:
            print(f"[*] Building dataset...")
            samples = {}
            failed = 0

            # Load camera intrinsics for each camera
            intrinsics = {}
            for s in self._serials:
                intr_file = os.path.join(
                    self._calib_dir,
                    "intrinsics",
                    f"{s}_{self._w}x{self._h}.yml",
                )
                with open(intr_file, "r") as f:
                    intr = yaml.load(f, Loader=yaml.FullLoader)
                intr = intr["color"]
                intrinsics[s] = np.array(
                    [
                        [intr["fx"], 0, intr["ppx"]],
                        [0, intr["fy"], intr["ppy"]],
                        [0, 0, 1],
                    ]
                )

            # Load all sequences
            for n in self._subjects:
                seq = [
                    os.path.join(n, s)
                    for s in sorted(os.listdir(os.path.join(self._root, n)))
                ]
                assert len(seq) == 100, "Incomplete sequences!"
                for i, q in enumerate(seq):
                    meta_file = os.path.join(self._root, q, "meta.yml")
                    with open(meta_file, "r") as f:
                        meta = yaml.load(f, Loader=yaml.FullLoader)
                    # # Fetch samples and compute labels for each camera
                    for c in self._serials:
                        for root, _, files in os.walk(os.path.join(self._root, q, c)):
                            for file in files:
                                if not file.startswith("color"):
                                    continue
                                idx = file.split("_")[-1].split(".")[0]
                                img_file = os.path.join(root, file)
                                with np.load(
                                    os.path.join(root, f"labels_{idx}.npz")
                                ) as labels:
                                    if np.all(labels["joint_3d"][0] == -1):
                                        failed += 1
                                        # continue
                                    obj_class_id = meta['ycb_ids'][meta["ycb_grasp_ind"]] - 1
                                    ho2d, ho3d = self._compute_labels(
                                        intrinsics[c], meta, labels, self._obj_labels[obj_class_id]
                                    )
                                    # # Rescale the 2D keypoints, because the images are rescaled from 640x480 to
                                    # # 224x224! This improves the performance of the 2D KP estimation GREATLY.
                                    ho2d[:, 0] = ho2d[:, 0] * 224.0 / self._w
                                    ho2d[:, 1] = ho2d[:, 1] * 224.0 / self._h
                                    if obj_class_id not in samples:
                                        samples[obj_class_id] = []
                                    samples[obj_class_id].append((img_file, ho2d, ho3d))
            if failed != 0:
                print(f"[!] {failed} samples were missing annotations!")
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving dataset into {pickle_path}...")
                pickle.dump(samples, pickle_file)

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
            kp2d_transform=None,
            object_as_task=object_as_task,
        )

        return dataset

    def _load(
        self,
        object_as_task: bool,
        split: str,
        shuffle: bool,
        normalize_keypoints: bool,
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        dataset = self._make_dataset(
            split,
            self._root,
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
                num_tasks=(
                    -1
                    if split == "train"
                    else (len(dataset) / (self.k_shots + self.n_queries))
                ),
            )
        else:
            split_dataset_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=8,
            )
        return split_dataset_loader
