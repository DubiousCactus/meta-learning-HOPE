#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from PIL import Image

import shutil
import os


whole_path = "/home/tmorales/RescaledFPHAD/"
file_root = os.path.join(whole_path, "Video_files")
size = (224, 224)

for root, dirs, files in os.walk(whole_path):
    if "object_pose.txt" in files:
        path = root.split(os.sep)
        subject, action_name, seq_idx = path[-3], path[-2], path[-1]
        video_seq = os.path.join(file_root, subject, action_name, seq_idx, "color")
        if not os.path.isdir(video_seq):
            continue
        for file_name in os.listdir(video_seq):
            img_path = os.path.join(video_seq, file_name)
            moved_path = os.path.join(video_seq, file_name + ".old")
            shutil.move(img_path, moved_path)
            img = Image.open(moved_path)
            new_img = img.resize(size)
            new_img.save(img_path)
            img.close()
            new_img.close()
            os.remove(moved_path)
