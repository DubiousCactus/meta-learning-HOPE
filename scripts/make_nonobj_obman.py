#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import os
import pickle

pickle_path, dest_path = (
    "/home/tmorales/Obman/train/obman_train_task.pkl",
    "/home/tmorales/Obman/train/obman_train.pkl",
)
with open(pickle_path, "rb") as pickle_file:
    samples = pickle.load(pickle_file)
    new_samples = []
    labels, i = {}, 0
    for k, v in samples.items():
        for img_path, p_2d, p_3d in v:
            new_samples.append((img_path, p_2d, p_3d))
    with open(dest_path, "wb") as new_pickle:
        pickle.dump(new_samples, new_pickle)
