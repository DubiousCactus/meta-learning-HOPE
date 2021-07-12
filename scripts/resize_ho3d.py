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


whole_path = "/home/tmorales/HO3D_v3/"
size = (224, 224)

file_roots = [
    os.path.join(whole_path, "train"),
    os.path.join(whole_path, "evaluation"),
]
for root in file_roots:
    for subject in os.listdir(root):
        s_path = os.path.join(root, subject)
        for img in os.listdir(os.path.join(s_path, "rgb")):
            img_path = os.path.join(s_path, "rgb", img)
            moved_path = os.path.join(s_path, "rgb", img + ".old")
            shutil.move(img_path, moved_path)
            img = Image.open(moved_path)
            new_img = img.resize(size)
            new_img.save(img_path)
            img.close()
            new_img.close()
            os.remove(moved_path)
