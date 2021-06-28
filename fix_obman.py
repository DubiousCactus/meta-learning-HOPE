#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import pickle

pickle_path, dest_path, root_path = (
        "/home/tmorales/Obman/train/obman_train_task.pkl.bak",
        "/home/tmorales/Obman/train/obman_train_task.pkl",
        "/home/tmorales/"
)
with open(pickle_path, "rb") as pickle_file:
    samples = pickle.load(pickle_file)
    shown = False
    new_samples = {}
    labels, i = {}, 0
    for k, v in samples.items():
        new_v = []
        for img_path, p_2d, p_3d in v:
            new_path = os.path.join(root_path, img_path[3:])
            if not shown:
                print(f"{img_path} -> {new_path}")
                shown = True
            new_v.append((new_path, p_2d, p_3d))
        new_samples[k] = new_v
    with open(dest_path, "wb") as new_pickle:
        pickle.dump(new_samples, new_pickle)
