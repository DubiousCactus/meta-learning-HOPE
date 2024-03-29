# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

from util.utils import fast_load_obj, compute_OBB_corners
from data.dataset.base import BaseDatasetTaskLoader
from typing import Union, Tuple, Optional
from torch.utils.data import DataLoader
from data.custom import CustomDataset
from functools import reduce
from tqdm import tqdm
from copy import copy

import learn2learn as l2l
import numpy as np
import itertools
import trimesh
import pickle
import torch
import yaml
import os


class NoInteractionError(Exception):
    def __init__(self):
        super().__init__()


class DexYCBDatasetTaskLoader(BaseDatasetTaskLoader):
    _subjects = [
        "20200709-subject-01",
        "20200813-subject-02",
        "20200820-subject-03",
        # "20200903-subject-04",
        # "20200908-subject-05",
        # "20200918-subject-06",
        # "20200928-subject-07",
        # "20201002-subject-08",
        # "20201015-subject-09",
        # "20201022-subject-10",
    ]

    _viewpoints = [
        "836212060125",
        "839512060362",
        "840412060917",
        "841412060263",
        "932122060857",
        "932122060861",
        "932122061900",
        "932122062010",
    ]

    obj_labels = [
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
        # "051_large_clamp", # Missing
        "052_extra_large_clamp",
        "061_foam_brick",
    ]

    def __init__(
        self,
        root: str,
        batch_size: int,
        k_shots: int,
        n_queries: int,
        tiny: bool = False,
        test: bool = False,
        object_as_task: bool = True,
        hold_out: int = 0,
        test_objects: Optional[int] = None,
        seed_factor: int = 1,
        normalize_keypoints: bool = False,
        hand_only: bool = True,
        use_cuda: bool = True,
        gpu_number: int = 0,
        auto_load: bool = True,  # In the analysis, we want to override the loading process
        interaction_threshold: float = 1e-4,
        min_fingers_interaction: int = 2,
    ):
        super().__init__(
            root,
            batch_size,
            k_shots,
            n_queries,
            test,
            object_as_task,
            normalize_keypoints,
            hand_only,
            use_cuda,
            gpu_number,
            auto_load=False,
        )
        self._calib_dir = os.path.join(self._root, "calibration")
        self._model_dir = os.path.join(self._root, "models")
        self._h, self._w = 480, 640
        self._bboxes = {}  # Cache
        self._ram_disk_path = "/dev/shm/DexYCB"
        self._interaction_threshold = interaction_threshold
        self._min_fingers_interaction = min_fingers_interaction

        # Don't use the base class autoloading, this is a custom loading. However we don't want
        # that either in the analysis script.
        if auto_load:
            self.split_categories = self._make_split_categories(
                hold_out, seed_factor=seed_factor
            )
            print(
                f"[*] Training with {', '.join([self.obj_labels[i] for i in self.split_categories['train']])}"
            )
            print(
                f"[*] Validating with {', '.join([self.obj_labels[i] for i in self.split_categories['val']])}"
            )
            print(
                f"[*] Testing with {', '.join([self.obj_labels[i] for i in self.split_categories['test']])}"
            )
            samples = self.make_raw_dataset(tiny=tiny)
            if test:
                if test_objects is not None:
                    """
                    This was added late in the experiments. It allows to keep the same splits defined
                    by the hold_out parameter, but to keep only test_objects from the test split (for
                    experiment 3).
                    """
                    print(f"Keepin only the first {test_objects} test objects")
                    del self.split_categories["test"][test_objects:]
                self.test = self._load(
                    samples,
                    object_as_task,
                    "test",
                    False,
                    normalize_keypoints,
                )
            else:
                self.train, self.val = self._load(
                    samples,
                    object_as_task,
                    "train",
                    True,
                    normalize_keypoints,
                ), self._load(
                    samples,
                    object_as_task,
                    "val",
                    False,
                    normalize_keypoints,
                )
            del samples

    def _make_split_categories(self, hold_out: int, seed_factor: int = 1) -> dict:
        """
        HO-3D contains 20 object categories. This method distributes those categories per split,
        according to "hold_out" which corresponds to how many are held out of the train split.
        However if N categories are held out, 2*N categories must be effectively held out because
        they can't be the same for the validation and test splits!

        seed_factor: Used to randomize experiment 3 (experiment 3 is just a sub experiment of
                     experiment 1 really... So it's experiment 3.b haha
        """
        categories = list(range(len(self.obj_labels)))
        idx = list(range(len(categories)))
        np.random.seed(hold_out * seed_factor)
        np.random.shuffle(idx)
        n_test, n_val = hold_out, min(5, hold_out // 2 + (hold_out % 2))
        n_train = len(self.obj_labels) - n_test - n_val
        assert n_train > 0, "There must remain at least one category in the train split"
        splits = {
            "train": list(np.array(categories)[idx[:n_train]]),
            "val": list(np.array(categories)[idx[n_train : n_train + n_val]]),
            "test": list(np.array(categories)[idx[-n_test:]]),
        }
        assert len(set(splits["train"]) & set(splits["val"]) & set(splits["test"])) == 0
        return splits

    def _compute_labels(
        self, cam_intr: np.ndarray, meta: dict, labels: dict, obj_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cache the mesh bounding box
        obj_file_path = os.path.join(
            self._root,
            "models",
            self.obj_labels[obj_id],
            "textured_simple.obj",
        )
        if obj_file_path not in self._bboxes:
            with open(obj_file_path, "r") as m_f:
                mesh = fast_load_obj(m_f)[0]
            mesh = trimesh.load(mesh)
            self._bboxes[obj_file_path] = compute_OBB_corners(mesh)

        vert3d = self._bboxes[obj_file_path]

        # Apply the rotation + translation to the bbox vertices
        # The format is [R; t] with R 3x3 and t 3x1.
        transform = labels["pose_y"][meta["ycb_grasp_ind"]]
        hom_verts = np.concatenate(
            [vert3d, np.ones([vert3d.shape[0], 1])],
            axis=1,
        )
        vert3d = transform.dot(hom_verts.T).T[:, :3]
        # For an interaction to be valid, there needs to be at least N fingertips inside the
        # bounding cuboid.
        centroid, obj_corner = np.mean(vert3d, axis=0), vert3d[0]
        finger_tip_distances = np.linalg.norm(
            np.tile(centroid, (5, 1)) - labels["joint_3d"][0, [4, 8, 12, 16, 20], :],
            axis=1,
        )
        corner_distance = np.linalg.norm(centroid - obj_corner)
        if (
            np.count_nonzero(
                (finger_tip_distances - corner_distance) < self._interaction_threshold
            )
            < self._min_fingers_interaction
        ):
            raise NoInteractionError
        # Project to 2D
        hom_2d_verts = np.dot(cam_intr, vert3d.transpose())
        vert2d = hom_2d_verts / hom_2d_verts[2, :]
        vert2d = vert2d[:2, :].transpose()
        return (
            torch.cat(
                [
                    torch.Tensor(labels["joint_2d"].reshape((21, 2))),
                    torch.Tensor(vert2d),
                ]
            ),
            torch.cat(
                [
                    torch.Tensor(labels["joint_3d"].reshape((21, 3)) * 1000),
                    torch.Tensor(vert3d) * 1000,
                ]
            ),
        )

    def make_raw_dataset(self, mirror_left_hand=False, tiny=False) -> dict:
        pickle_path = (
            os.path.join(self._root, "dexycb_tiny.pkl" if tiny else "dexycb.pkl")
            if not mirror_left_hand
            else os.path.join(self._root, f"dexycb_mirrorred.pkl")
        )
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                print(f"[*] Loading dataset from {pickle_path}...")
                samples = pickle.load(pickle_file)
        else:
            print(f"[*] Building dataset...")
            if tiny:
                self._subjects = self._subjects[:3]
                self._viewpoints = self._viewpoints[:3]
            pbar = tqdm(total=len(self._subjects) * len(self._viewpoints) * 100)
            samples = {}
            failed, no_interaction = 0, {i: 0 for i in range(len(self.obj_labels))}

            # Load camera intrinsics for each camera
            intrinsics = {}
            for s in self._viewpoints:
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
            for subject_id in self._subjects:
                seq = [
                    os.path.join(subject_id, s)
                    for s in sorted(os.listdir(os.path.join(self._root, subject_id)))
                ]
                assert len(seq) == 100, "Incomplete sequences!"
                for i, sequence_id in enumerate(seq):
                    meta_file = os.path.join(self._root, sequence_id, "meta.yml")
                    with open(meta_file, "r") as f:
                        meta = yaml.load(f, Loader=yaml.FullLoader)
                    # Fetch samples and compute labels for each camera
                    for c in self._viewpoints:
                        for root, _, files in os.walk(
                            os.path.join(self._root, sequence_id, c)
                        ):
                            for file in files:
                                if not file.startswith("color"):
                                    continue
                                idx = file.split("_")[-1].split(".")[0]
                                img_file = os.path.join(root, file)
                                if img_file.endswith(".old"):
                                    continue
                                with np.load(
                                    os.path.join(root, f"labels_{idx}.npz")
                                ) as labels:
                                    obj_class_id = (
                                        meta["ycb_ids"][meta["ycb_grasp_ind"]] - 1
                                    )
                                    if obj_class_id == 18:
                                        # Large clamp missing!
                                        continue
                                    elif obj_class_id > 18:
                                        obj_class_id -= 1  # Compensate for the one removed in the list
                                    if np.all(
                                        labels["joint_3d"].reshape((21, 3)) == -1
                                    ):
                                        failed += 1
                                        continue
                                    try:
                                        ho2d, ho3d = self._compute_labels(
                                            intrinsics[c],
                                            meta,
                                            labels,
                                            obj_class_id,
                                        )
                                        if (
                                            mirror_left_hand
                                            and meta["mano_sides"][0].lower() == "left"
                                        ):
                                            ho3d = (
                                                -ho3d
                                            )  # Simple reflection through a hyperplane
                                    except NoInteractionError:
                                        no_interaction[obj_class_id] += 1
                                        continue
                                    # # Rescale the 2D keypoints, because the images are rescaled from 640x480 to
                                    # # 224x224! This improves the performance of the 2D KP estimation GREATLY.
                                    # TODO: Properly rescale the 2D labels
                                    # First I resized the images to 256x256, then I center cropped
                                    # to 224x224
                                    # ho2d[:, 0] = ho2d[:, 0] * 256.0 / self._w
                                    # ho2d[:, 1] = ho2d[:, 1] * 256.0 / self._h

                                    # Use a combination of object and subject IDs so that a tasks
                                    # constitues the manipulation of one object by one subject
                                    # only. This is necessary to avoid including different grasps
                                    # caused by different subjects!
                                    key = (obj_class_id, sequence_id)
                                    if key not in samples:
                                        samples[key] = []
                                    samples[key].append((img_file, ho2d, ho3d))
                        pbar.update()
            if failed != 0:
                print(f"[!] {failed} samples were missing annotations!")
            for id, val in no_interaction.items():
                if id in samples.keys():
                    print(
                        f"[!] {val} samples with no interaction were removed from {self.obj_labels[id]}: {len(samples[id])} samples left"
                    )
            with open(pickle_path, "wb") as pickle_file:
                print(f"[*] Saving dataset into {pickle_path}...")
                pickle.dump(samples, pickle_file)
        return samples

    def make_dataset(
        self,
        split: str,
        samples: dict,
        object_as_task=False,
        normalize_keypoints=False,
    ) -> CustomDataset:
        # Hold out
        keys = list(samples.copy().keys())
        for category_id, sequence_id in keys:
            if category_id not in self.split_categories[split]:
                del samples[(category_id, sequence_id)]

        print(
            f"[*] Loaded {reduce(lambda x, y: x + y, [len(x) for x in samples.values()])} samples from the {split} split."
        )
        print(f"[*] Total unique tasks (object/subject): {len(samples.keys())}")
        if not object_as_task:  # Transform to list
            samples = list(itertools.chain.from_iterable(samples.values()))
        print(f"[*] Generating dataset in pinned memory...")
        to_ramdisk = lambda path: os.path.join(
            self._ram_disk_path, path.split("DexYCB/")[1]
        )
        dataset = CustomDataset(
            samples,
            img_transform=self._img_transform,
            kp2d_transform=None,
            object_as_task=object_as_task,
            hand_only=self._hand_only,
            to_ram_disk_fn=to_ramdisk if os.path.isdir(self._ram_disk_path) else None,
        )

        return dataset

    def _load(
        self,
        samples: dict,
        object_as_task: bool,
        split: str,
        shuffle: bool,
        normalize_keypoints: bool,
    ) -> Union[DataLoader, l2l.data.TaskDataset]:
        dataset = self.make_dataset(
            split,
            copy(samples),  # They will be modified, we'll need them for the other split
            object_as_task=object_as_task,
            normalize_keypoints=normalize_keypoints,
        )
        if object_as_task:
            assert dataset.min_task_samples >= (
                self.k_shots + self.n_queries
            ), f"At least one task has only {dataset.min_task_samples} samples, which is less than < (K + Q)! Consider using different K and Q paramters."
            split_dataset = l2l.data.MetaDataset(
                dataset, indices_to_labels=dataset.class_labels
            )
            split_dataset_loader = l2l.data.TaskDataset(
                split_dataset,
                [
                    l2l.data.transforms.FusedNWaysKShots(
                        split_dataset,
                        n=1,
                        k=self.k_shots + self.n_queries,
                        replacement=False,
                    ),
                    l2l.data.transforms.LoadData(split_dataset),
                ],
                num_tasks=(
                    -1
                    if split == "train"
                    else (len(dataset) // (self.k_shots + self.n_queries))
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
