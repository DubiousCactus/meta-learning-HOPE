#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Procrustes Analysis of the DexYCB tasks dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import hydra
import os

from abc import abstractclassmethod, ABC
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from typing import List
from copy import copy

from util.factory import DatasetFactory
from data.custom import CustomDataset
from util.utils import plot_pose


class ProcrustesAnalysis(ABC):
    @abstractclassmethod
    def centroid(self, x):
        """
        Compute the shape centroid.
        """
        return np.mean(x, axis=0)

    @abstractclassmethod
    def centroid_size(self, x, centroid):
        return np.sum(np.sqrt(np.square(x - centroid)))

    @abstractclassmethod
    def frobenius_norm_size(self, x, centroid):
        return np.sqrt(np.sum(np.square(x - centroid)))

    @abstractclassmethod
    def align_shapes(self, x1, x2, shape_size_metric="frobenius_norm"):
        """
        Align both shapes by superimposing x1 upon x2 (the reference object).
        """
        x1_centroid = self.centroid(x1)
        x2_centroid = self.centroid(x2)

        # Re-scale each shape to have equal size
        if shape_size_metric == "frobenius_norm":
            x1_size = self.frobenius_norm_size(x1, x1_centroid)
            x2_size = self.frobenius_norm_size(x2, x2_centroid)
        elif shape_size_metric == "centroid":
            x1_size = self.centroid_size(x1, x1_centroid)
            x2_size = self.centroid_size(x2, x2_centroid)
        else:
            raise ValueError(f"{shape_size_metric} is not a valid shape size metric!")

        # Align w.r.t. position the two shapes at their centroids, and uniformly scale them.
        x1_ = (x1 - x1_centroid) / x1_size
        x2_ = (x2 - x2_centroid) / x2_size

        # Align w.r.t. orientation by rotation
        u, s, vh = np.linalg.svd(x1_.T @ x2_)
        rot_mat = vh.T @ u.T
        # Superimpose x1 upon x2
        x1_ = (rot_mat @ x1_.T).T
        return x1_, x2_

    @abstractclassmethod
    def compute_distance(self, x1, x2):
        """
        Compute the Procrustes Shape Distance between shapes x1 and x2.
        """
        x1_, x2_ = self.align_shapes(x1, x2)
        return np.sum(np.square(x1_ - x2_))

    @abstractclassmethod
    def generalised_procrustes_analysis(self, X) -> np.ndarray:
        """
        Compute an estimate of the mean shape of the set of shapes X.
        """
        mean, delta, iter = X[0], float("+inf"), 0
        while delta > 1e-13 and iter < 30:
            # TODO: Properly fix the size and orientation by normalisation to avoid shrinking or
            # drifting of the mean shape (isn't that done in align_shapes??)
            for i in range(len(X)):
                aligned_shape, _ = self.align_shapes(X[i], mean)
                X[i] = aligned_shape
            # The Procrustes mean is the mean of the aligned shapes
            new_mean = np.mean(np.array(X), axis=0)
            delta = ProcrustesAnalysis.compute_distance(mean, new_mean)
            mean = new_mean
            iter += 1
        return mean


# =========== Synthetic data (triangles) to test and debug ProcrustesAnalysis =============
def build_synthetic_set(n=5) -> List[np.ndarray]:
    X = []
    rng = np.random.default_rng()
    for _ in range(n):
        X.append(rng.random((3, 2)))
    return X


def display_synth(X, prompt=False):
    W, H = 640, 480
    # Black canvas:
    img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    img_draw = ImageDraw.Draw(img)
    rng = np.random.default_rng()
    for i in range(len(X)):
        t = X[i]
        t_ = t.copy()
        color = tuple(rng.integers(0, 255, size=3, endpoint=True))
        # Resize t
        for i in range(t.shape[0]):
            t_[i][0] *= W
            t_[i][1] *= H
        img_draw.line(((t_[0][0], t_[0][1]), (t_[1][0], t_[1][1])), fill=color)
        img_draw.line(((t_[1][0], t_[1][1]), (t_[2][0], t_[2][1])), fill=color)
        img_draw.line(((t_[2][0], t_[2][1]), (t_[0][0], t_[0][1])), fill=color)
    img.show()
    if prompt:
        print("Save dataset? [N/y]: ")
        a = input()
        if a == "y":
            with open("dummy.pkl", "wb") as f:
                pickler = pickle.Pickler(f)
                pickler.dump(X)


def main_debug():
    # I may want to use https://procrustes.readthedocs.io
    loaded = False
    if os.path.isfile("dummy.pkl"):
        with open("dummy.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            X = unpickler.load()
            loaded = True
    else:
        X = build_synthetic_set()
    display_synth(X, prompt=not loaded)
    print("[*] Applying Generalised Procrustes Analysis...")
    ProcrustesAnalysis.generalised_procrustes_analysis(X)
    # There seems to have been a scale drift... Let's fix it:
    for i in range(len(X)):
        X[i] *= 0.5
    # Now X is centered at the origin, so to visualise it we may shift all objects to the center of
    # the frame, like so:
    for i in range(len(X)):
        X[i] += np.array([0.5, 0.5])
    display_synth(X)


# ======================================================================================


def plot_3D_hands(pose1, title1, pose2, title2):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    plot_pose(ax1, pose1, plot_obj=False)
    ax1.set_title(title1)
    ax2 = fig.add_subplot(122, projection="3d")
    plot_pose(ax2, pose2, plot_obj=False)
    ax2.set_title(title2)
    plt.show()


def check_overlap(cfg: DictConfig):
    # TODO: Remove this. The bug has been fixed and models retrained.
    ##################### Checking overlap due to the bug in the objects pruning #############################
    from functools import reduce

    level_splits = {}
    for hold_out in range(5, 14):
        print(f"[*] Computing overlap for hold_out={hold_out}")
        cfg.experiment.hold_out = hold_out
        dataset_loader = DatasetFactory.make_data_loader(
            cfg,
            to_absolute_path(cfg.shapenet_root),
            cfg.experiment.dataset,
            to_absolute_path(cfg.experiment.dataset_path),
            cfg.experiment.batch_size,
            cfg.test_mode,
            cfg.experiment.k_shots,
            cfg.experiment.n_queries,
            hand_only=True,
            object_as_task=cfg.experiment.object_as_task,
            normalize_keypoints=cfg.experiment.normalize_keypoints,
            augment_fphad=cfg.experiment.augment,
            auto_load=False,
        )
        original_samples = dataset_loader.make_raw_dataset(mirror_left_hand=True)

        split_objects, expected_splits = {}, {}
        for split in ["train", "val", "test"]:
            samples = copy(original_samples)
            expected_samples = copy(original_samples)
            keys = list(samples.copy().keys())
            for category_id in keys:
                if category_id not in dataset_loader.split_categories[split]:
                    del samples[keys[category_id]]
                    del expected_samples[category_id]
            split_objects[split] = list(samples.keys())
            expected_splits[split] = list(expected_samples.keys())
        print("Actual splits: ", split_objects)
        print("Expected splits: ", expected_splits)
        level_splits[hold_out] = (split_objects, expected_splits)
        val_in_train = reduce(
            lambda a, b: a + b,
            [i for i in split_objects["val"] if i in split_objects["train"]] + [0],
        )
        test_in_train = reduce(
            lambda a, b: a + b,
            [i for i in split_objects["test"] if i in split_objects["train"]] + [0],
        )
        test_in_val = reduce(
            lambda a, b: a + b,
            [i for i in split_objects["test"] if i in split_objects["val"]] + [0],
        )
        print(f"Validation objects in train set: {val_in_train}")
        print(f"Test objects in train set: {test_in_train}")
        print(f"Test objects in validation set: {test_in_val}")
    prev_actual, prev_expected = [], []
    for lvl, splits in level_splits.items():
        actual, expected = splits[0]["test"], splits[1]["test"]
        test_overlap_actual = reduce(lambda a, b: a+b, [1 for i in actual if i in prev_actual] + [0])
        test_overlap_expected = reduce(lambda a, b: a+b, [1 for i in expected if i in prev_expected] + [0])
        print(f"[*] Overlap from {lvl} to {lvl-1} in actual test splits: {test_overlap_actual}")
        print(f"[*] Overlap from {lvl} to {lvl-1} in expected test splits: {test_overlap_expected}")
        prev_actual, prev_expected = actual, expected


def compute_test_to_mean_train(samples, dataset_loader):
    train_set = dataset_loader.make_dataset("train", copy(samples), object_as_task=True)
    poses = [pose3d.cpu().numpy() for _, _, pose3d in train_set]

    train_samples = copy(samples)
    keys = list(train_samples.copy().keys())
    for category_id in keys:
        if category_id not in dataset_loader.split_categories["train"]:
            del train_samples[keys[category_id]]

    print("\n\n\n\n--------------------------------------------------")
    print(f"[*] Training set objects: {', '.join([dataset_loader.obj_labels[i] for i in train_samples.keys()])}")
    print("[*] Computing mean train shape...")
    mean_train_pose = ProcrustesAnalysis.generalised_procrustes_analysis(poses)

    # Build one dataset per task for the analysis of the test set
    mean_test_dist, total_test_poses = .0, 0
    test_samples = copy(samples)
    keys = list(test_samples.copy().keys())
    for category_id in keys:
        if category_id not in dataset_loader.split_categories["test"]:
            del test_samples[keys[category_id]]
    for obj_id, task in test_samples.items():
        print(f"[*] Computing the mean distance of {dataset_loader.obj_labels[obj_id]} to the mean train shape...")
        # Still using the custom dataset because of the preprocessing (root alignment)
        # Set object_as_task=False because we pass it a list and not a dict
        task_dataset = CustomDataset(task, object_as_task=False)
        poses = [pose3d.cpu().numpy() for _, _, pose3d in task_dataset]
        distances = []
        for pose in poses:
            distances.append(ProcrustesAnalysis.compute_distance(pose, mean_train_pose))
            total_test_poses += 1
        distances = np.array(distances)
        print(f"[*] Mean distance to the mean train shape (MSE): {np.mean(distances)}")
        mean_test_dist += len(poses) * np.mean(distances) # Weighted average!
        if cfg.vis:
            fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
            ax.hist(distances)
            ax.set_title("Histogram of Procrustes distances to the mean train shape")
            fig.show()
            mean_task_pose = ProcrustesAnalysis.generalised_procrustes_analysis(poses)
            plot_3D_hands(
                mean_train_pose,
                "Mean train pose",
                mean_task_pose,
                f"Mean {dataset_loader.obj_labels[obj_id]} pose",
            )
            print(
                f"[*] Distance of mean task shape to the mean train shape: \
            {ProcrustesAnalysis.compute_distance(mean_task_pose, mean_train_pose)}"
            )
    print(f"[*] Mean test-train distance: {mean_test_dist/total_test_poses}")

def compute_dist_matrix(samples, dataloader):
    task_mean_poses = {}
    obj_tasks = {}
    for (obj_id, sequence_id), task in samples.items():
        if obj_id not in obj_tasks:
            obj_tasks[obj_id] = []
        obj_tasks[obj_id] += task
    for obj_id, task in obj_tasks.items():
        print(f"[*] Computing the mean shape of {dataloader.obj_labels[obj_id]}...")
        # Still using the custom dataset because of the preprocessing (root alignment)
        # Set object_as_task=False because we pass it a list and not a dict
        task_dataset = CustomDataset(task, object_as_task=False)
        poses = [pose3d.cpu().numpy() for _, _, pose3d in task_dataset]
        task_mean_poses[obj_id] = ProcrustesAnalysis.generalised_procrustes_analysis(poses)
    print("[*] Computing distances...")
    lines = []
    print(f"_________________|{''.join(['{s:{c}^{n}}|'.format(s=dataloader.obj_labels[i][4:], n=17, c=' ') for i in task_mean_poses.keys()])}")
    lines.append(f"_________________|{''.join(['{s:{c}^{n}}|'.format(s=dataloader.obj_labels[i][4:], n=17, c=' ') for i in task_mean_poses.keys()])}\n")
    for obj_id, pose_a in task_mean_poses.items():
        print("{s:{c}^{n}}|".format(s=dataloader.obj_labels[obj_id][4:], n=17, c=' '), end="", flush=False)
        line = "{s:{c}^{n}}|".format(s=dataloader.obj_labels[obj_id][4:], n=17, c=' ')
        for obj_id, pose_b in task_mean_poses.items():
            dist = ProcrustesAnalysis.compute_distance(pose_a, pose_b)
            print("{s:{c}^{n}}|".format(s=f"{dist:.4f}", n=17, c=' '), end="", flush=False)
            line += "{s:{c}^{n}}|".format(s=f"{dist:.4f}", n=17, c=' ')
        print()
        lines.append(line+"\n")
    with open("distance_matrix.txt", "w") as f:
        f.writelines(lines)
        print(f"[*] Saved distance matrix in {os.path.join(os.getcwd(), 'distance_matrix.txt')}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataset_loader = DatasetFactory.make_data_loader(
        cfg,
        to_absolute_path(cfg.shapenet_root),
        cfg.experiment.dataset,
        to_absolute_path(cfg.experiment.dataset_path),
        cfg.experiment.batch_size,
        cfg.test_mode,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        hand_only=True,
        object_as_task=cfg.experiment.object_as_task,
        normalize_keypoints=cfg.experiment.normalize_keypoints,
        augment_fphad=cfg.experiment.augment,
        auto_load=False,
    )
    samples = dataset_loader.make_raw_dataset(mirror_left_hand=True)
    compute_dist_matrix(samples, dataset_loader)
    # check_overlap(cfg)


if __name__ == "__main__":
    main()
