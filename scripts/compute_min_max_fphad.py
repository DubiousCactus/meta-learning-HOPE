#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import os


whole_path = "/home/cactus/Code/FPHAD/"
file_root = os.path.join(whole_path, "Video_files")

for root, dirs, files in os.walk(whole_path):
    if "object_pose.txt" in files:
        path = root.split(os.sep)
        subject, action_name, seq_idx = path[-3], path[-2], path[-1]
        video_seq = os.path.join(file_root, subject, action_name, seq_idx, "color")
        if not os.path.isdir(video_seq):
            continue
        for file_name in os.listdir(video_seq):
