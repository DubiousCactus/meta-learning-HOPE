#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utility functions for data loading and processing.
"""

from collections import OrderedDict
from PIL import ImageDraw

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import pickle
import torch
import os


def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, "decode"):
        text = text.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n") + " \n"

    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current["f"]) > 0:
            # get vertices as clean numpy array
            vertices = np.array(current["v"], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current["f"], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (
                np.array(list(remap.keys())),
                np.array(list(remap.values())),
            )
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                "vertices": vertices[vert_order],
                "faces": face_order[faces],
                "metadata": {},
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current["g"]) > 0:
                face_groups = np.zeros(len(current["f"]) // 3, dtype=np.int64)
                for idx, start_f in current["g"]:
                    face_groups[start_f:] = idx
                loaded["metadata"]["face_groups"] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ["v"]}
    current = {k: [] for k in ["v", "f", "g"]}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == "f":
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split("/")
                    current["v"].append(attribs["v"][int(f_split[0]) - 1])
                current["f"].append(remap[f])
        elif line_split[0] == "o":
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == "g":
            # defining a new group
            group_idx += 1
            current["g"].append((group_idx, len(current["f"]) // 3))

    if next_idx > 0:
        append_mesh()

    del text
    file_obj.close()

    return meshes


def load_state_dict(module, resnet_path):
    """
    Load a state_dict in a module agnostic way (when DataParallel is used, the module is
    wrapped and the saved state dict is not applicable to non-wrapped modules).
    """
    ckpt = torch.load(resnet_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt["model_state_dict"].items():
        if "module" in k:
            k = k.replace("module.", "")
        new_state_dict[k] = v
    module.load_state_dict(new_state_dict)


def draw_2D_prediction(image, pred):
    draw = ImageDraw.Draw(image)
    # Object bbox:
    obj_pred = pred[21:, :]
    """
    draw.line([tuple(obj_pred[0, :]), tuple(obj_pred[1, :])], fill="red", width=0) # 1, 2
    draw.line([tuple(obj_pred[2, :]), tuple(obj_pred[3, :])], fill="red", width=0) # 5, 6
    draw.line([tuple(obj_pred[4, :]), tuple(obj_pred[5, :])], fill="red", width=0) # 7, 8
    draw.line([tuple(obj_pred[6, :]), tuple(obj_pred[7, :])], fill="red", width=0) # 3, 4
    draw.line([tuple(obj_pred[0, :]), tuple(obj_pred[6, :])], fill="red", width=0) # 1, 3
    draw.line([tuple(obj_pred[1, :]), tuple(obj_pred[7, :])], fill="red", width=0) # 2, 4
    draw.line([tuple(obj_pred[2, :]), tuple(obj_pred[4, :])], fill="red", width=0) # 5, 7
    draw.line([tuple(obj_pred[3, :]), tuple(obj_pred[5, :])], fill="red", width=0) # 6, 8

    draw.line([tuple(obj_pred[0, :]), tuple(obj_pred[2, :])], fill="red", width=0) # 1, 5
    draw.line([tuple(obj_pred[1, :]), tuple(obj_pred[3, :])], fill="red", width=0) # 2, 6
    draw.line([tuple(obj_pred[7, :]), tuple(obj_pred[5, :])], fill="red", width=0) # 4, 8
    draw.line([tuple(obj_pred[6, :]), tuple(obj_pred[4, :])], fill="red", width=0) # 3, 7
    # Hand vertices:
    hand_pred = pred[:21, :]
    draw.line([tuple(hand_pred[0, :]), tuple(hand_pred[2, :])], fill="red", width=0) # 1, 5
    """



def plot_3D_pred_gt(pred, gt=None):
    def plot_pred(ax, pred):
        x = np.linspace(-1, 1, 200)
        pred /= 1000  # From millimeters
        # Plot the hand first:
        prev = pred[0, :]
        for row in np.ndindex(pred.shape[0] - 8):
            cur = pred[row, :]
            if row[0] in [5, 9, 13, 17]:
                prev = pred[0, :]
            cur, prev = cur.flatten(), prev.flatten()
            x, y, z = (
                np.linspace(prev[0], cur[0], 100),
                np.linspace(prev[1], cur[1], 100),
                np.linspace(prev[2], cur[2], 100),
            )
            ax.plot(x, y, z, color="red")
            ax.text(cur[0], cur[1], cur[2], f"{row[0]}", color="red")
            prev = cur
        # Plot the object bbox:
        faces = [
            [0, 1, 5, 4, 0],
            [0, 1, 3, 2, 0],
            [0, 2, 6, 4, 0],
            [1, 5, 7, 3, 1],
            [2, 3, 7, 6, 2],
            [5, 7, 6, 4, 5]
        ]
        for face in faces:
            prev = pred[21+face[0], :]
            for idx in face:
                row = 21 + idx
                cur = pred[row, :]
                cur, prev = cur.flatten(), prev.flatten()
                x, y, z = (
                    np.linspace(prev[0], cur[0], 100),
                    np.linspace(prev[1], cur[1], 100),
                    np.linspace(prev[2], cur[2], 100),
                )
                ax.plot(x, y, z, color="green")
                ax.text(cur[0], cur[1], cur[2], f"{row-21}", color="green")
                prev = cur
    if gt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
        plot_pred(ax1, pred)
        ax1.set_title("Prediction")
        plot_pred(ax2, gt)
        ax2.set_title("Ground truth")
    else:
        ax = plt.figure().add_subplot(projection="3d")
        plot_pred(ax, pred)
    plt.show()
