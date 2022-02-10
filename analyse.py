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

import numpy as np
import pickle
import hydra
import os

from util.factory import DatasetFactory, AlgorithmFactory
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from abc import abstractclassmethod, ABC
from util.utils import plot_3D_hand
from PIL import Image, ImageDraw
from typing import List
from copy import copy


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
        for i in range(x1_.shape[0]):
            x1_[i] = rot_mat @ x1_[i]
        x1__ = x1_ @ rot_mat  # TODO: Work out the math to see if this is always correct
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
        while delta > 1e-12 or iter > 100:
            # TODO: Properly fix the size and orientation by normalisation to avoid shrinking or
            # drifting of the mean shape (isn't that done in align_shapes??)
            for i in range(len(X)):
                aligned_shape, _ = self.align_shapes(X[i], mean)
                X[i] = aligned_shape
            # The Procrustes mean is the mean of the aligned shapes
            new_mean = np.mean(np.array(X), axis=0)
            delta = ProcrustesAnalysis.compute_distance(mean, new_mean)
            print(f"-> Delta={delta}")
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
        object_as_task=cfg.experiment.object_as_task,
        normalize_keypoints=cfg.experiment.normalize_keypoints,
        augment_fphad=cfg.experiment.augment,
        auto_load=False,
    )
    samples = dataset_loader.make_raw_dataset(mirror_left_hand=True)
    train_set = dataset_loader.make_dataset("train", copy(samples), object_as_task=True)
    poses = [pose3d.cpu().numpy() for _, _, pose3d in train_set]
    # plot_3D_hand(poses[0])
    mean_train_pose = ProcrustesAnalysis.generalised_procrustes_analysis(poses)
    plot_3D_hand(mean_train_pose)
    # for sample in train_set:
    # img, pose_2d, pose_3d = sample
    # plot_3D_hand(pose_3d)
    # break


if __name__ == "__main__":
    main()
