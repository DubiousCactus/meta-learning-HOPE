#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Compute group statistics for a given dataset
"""

from sklearn.manifold import TSNE
from functools import reduce
from tqdm import tqdm
from PIL import Image

import pandas as pd
import numpy as np
import random
import pickle
import torch
import os

ho3d_obj_labels = [
    "011_banana",
    "010_potted_meat_can",
    "006_mustard_bottle",
    "004_sugar_box",
    "035_power_drill",
    "037_scissors",
    "021_bleach_cleanser",
    "025_mug",
    "003_cracker_box",
    "019_pitcher_base",
]

dataset_pickle_file = "ho3d.pkl"
dataset_root = "../HO3D_v3"
obj_labels = ho3d_obj_labels
batch_size = 64
tsne_n_samples = 1024


def stats(samples):
    stats = {}
    for group_id in samples.keys():
        print(f"[*] Computing statistics for {obj_labels[group_id]}...")
        img_channels_sum, img_channels_squared_sum, labels_sum, labels_squared_sum = (
            0,
            0,
            0,
            0,
        )
        group_samples = samples[group_id]
        num_batches = len(group_samples) // batch_size + 1
        label0 = group_samples[0][1].numpy()
        img_w, img_h = 224, 224
        pixel_count = len(group_samples) * img_w * img_h
        joints_count = len(group_samples) * label0.shape[0]
        print(
            f"Samples: {len(group_samples)} - Batches: {num_batches} - WxH: {img_w}x{img_h}"
        )
        # for batch_no in tqdm(range(num_batches)):
        # batch = group_samples[batch_no * batch_size : min((batch_no + 1) * batch_size, len(group_samples)-1)]
        # img_batch = np.array([np.array(Image.open(s[0])) for s in batch])
        # label_batch = np.zeros(
        # (batch_size, batch[0][1].shape[0], batch[0][1].shape[1])
        # )
        # for i, s in enumerate(batch):
        # label_batch[i, :, :] = s[1].numpy()
        # # Mean over batch, height and width, but not over the channels
        # img_channels_sum += np.mean(img_batch, axis=(0, 1, 2))
        # img_channels_squared_sum += np.mean(np.square(img_batch), axis=(0, 1, 2))
        # labels_sum += np.mean(label_batch, axis=(0, 1))
        # labels_squared_sum += np.mean(np.square(label_batch), axis=(0, 1))

        # img_mean = img_channels_sum / num_batches
        # img_squared_mean = img_channels_squared_sum / num_batches
        # label_mean = labels_sum / num_batches

        # # std = sqrt(E[X^2] - (E[X])^2)
        # print(img_mean, img_squared_mean, img_mean ** 2)
        # print(img_squared_mean - np.square(img_mean))
        # img_std = np.sqrt((img_squared_mean - np.square(img_mean)))
        # label_std = (labels_squared_sum / num_batches - label_mean ** 2) ** 0.5
        # stats[group_id] = img_std, label_std
        for img_path, label, _ in group_samples:
            img = np.array(Image.open(img_path).resize((img_w, img_h)))
            img_channels_sum += img.sum(axis=(0, 1))
            labels_sum += label.numpy().sum(axis=0)

        img_mean = img_channels_sum / pixel_count
        label_mean = labels_sum / joints_count

        img_channels_sum, labels_sum = 0, 0
        for img_path, labels, _ in group_samples:
            img = np.array(Image.open(img_path).resize((img_w, img_h)))
            img_channels_sum += ((img - img_mean) ** 2).sum(axis=(0, 1))
            labels_sum += ((label.numpy() - label_mean) ** 2).sum(axis=0)
        # var_sum = 0
        # for batch_no in tqdm(range(num_batches)):
        # batch = group_samples[batch_no * batch_size : min((batch_no + 1) * batch_size, len(group_samples)-1)]
        # img_batch = np.array([np.array(Image.open(s[0])) for s in batch])
        # label_batch = np.zeros(
        # (batch_size, batch[0][1].shape[0], batch[0][1].shape[1])
        # )
        # for i, s in enumerate(batch):
        # label_batch[i, :, :] = s[1].numpy()
        #     var_sum += np.square(img_batch - img_mean).sum(axis=(0, 1, 2))

        # img_var = (img_channels_squared_sum / pixel_count) - (img_mean**2)
        img_var = img_channels_sum / pixel_count
        label_var = labels_sum / joints_count
        img_std = np.sqrt(img_var)
        label_std = np.sqrt(label_var)
        stats[group_id] = {'img': (img_mean, img_std), 'lbl': (label_mean, label_std)}

        print(f"\t-> Images: mean={img_mean}, std={img_std}")
        print(f"\t-> Labels: {label_mean}, {label_std}")

    print(f"[*] Saving statistics as stats.pkl...")
    with open("stats.pkl", "wb") as file:
        pickle.dump(stats, file)


def tsne(samples):
    img_embeddings_df = {"Object": [], "X": [], "Y": []}
    lbl_embeddings_df = {"Object": [], "X": [], "Y": []}
    for group_id in samples.keys():
        print(f"[*] Computing t-SNE embeddings for {obj_labels[group_id]}...")
        group_samples = (
            random.sample(samples[group_id], tsne_n_samples)
            if len(samples[group_id]) > tsne_n_samples
            else samples[group_id]
        )
        images = np.array(
            [
                np.array(Image.open(s[0]).resize((224, 224))).flatten()
                for s in group_samples
            ]
        )
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(images)
        for x in X_embedded:
            img_embeddings_df["Object"].append(group_id)
            img_embeddings_df["X"].append(x[0])
            img_embeddings_df["Y"].append(x[1])
        del X_embedded, images

        labels = np.array([s[1].numpy().flatten() for s in group_samples])
        Y_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(labels)

        for y in Y_embedded:
            lbl_embeddings_df["Object"].append(group_id)
            lbl_embeddings_df["X"].append(y[0])
            lbl_embeddings_df["Y"].append(y[1])

    embeddings_df = {
        "img": pd.DataFrame(data=img_embeddings_df),
        "lbl": pd.DataFrame(data=lbl_embeddings_df),
    }
    with open("tsne.pkl", "wb") as file:
        pickle.dump(embeddings_df, file)


if __name__ == "__main__":
    pickle_path = os.path.join(dataset_root, f"{dataset_pickle_file}")
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            print(f"[*] Loading dataset from {pickle_path}...")
            samples = pickle.load(pickle_file)
    else:
        print(f"[!] Dataset picle file missing: {dataset_pickle_file}")
        exit(1)

    print(
        f"[*] Loaded {reduce(lambda x, y: x + y, [len(x) for x in samples.values()])} samples from the entire dataset."
    )
    print(f"[*] Total object categories: {len(samples.keys())}")

    stats(samples)
    # tsne(samples)
