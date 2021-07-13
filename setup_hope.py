#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Setup instructions to package the HOPE sub-project.
"""

from distutils.core import setup

import os


os.chdir("HOPE")
for mod in ["", "models", "utils"]:
    try:
        os.mknod(os.path.join(mod, "__init__.py"))
    except:
        pass

setup(
    name="HopeNet",
    version="1.1",
    py_modules=[
        "utils.dataset",
        "utils.model",
        "utils.options",
        "models.graphunet",
        "models.hopenet",
        "models.resnet",
    ],
)

os.chdir("../")
try:
    os.symlink("./HOPE/datasets", "datasets")
except Exception as e:
    print(f"[!] Could not create symlink: \n\t -> {e}")
