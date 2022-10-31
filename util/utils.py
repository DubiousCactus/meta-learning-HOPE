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

from model.cnn import ResNet, ResNet12, MobileNet

from collections import OrderedDict
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import torch


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


def plot_pose(ax, pose, plot_obj=True):
    # Plot the hand first:
    prev = pose[0, :]
    for row in np.ndindex(pose.shape[0] - 8):
        cur = pose[row, :]
        if row[0] in [5, 9, 13, 17]:
            prev = pose[0, :]
        cur, prev = cur.flatten(), prev.flatten()
        x, y, z = (
            np.linspace(prev[0], cur[0], 100),
            np.linspace(prev[1], cur[1], 100),
            np.linspace(prev[2], cur[2], 100),
        )
        ax.plot(x, y, z, color="red")
        ax.text(cur[0], cur[1], cur[2], f"{row[0]}", color="red")
        prev = cur
    if plot_obj:
        # Plot the object bbox:
        faces = [
            [0, 1, 2, 3, 0],
            [0, 1, 5, 4, 0],
            [0, 3, 7, 4, 0],
            [1, 5, 6, 2, 1],
            [2, 3, 7, 6, 2],
            [5, 6, 7, 4, 5],
        ]
        for face in faces:
            prev = pose[21 + face[0], :]
            for idx in face:
                row = 21 + idx
                cur = pose[row, :]
                cur, prev = cur.flatten(), prev.flatten()
                x, y, z = (
                    np.linspace(prev[0], cur[0], 100),
                    np.linspace(prev[1], cur[1], 100),
                    np.linspace(prev[2], cur[2], 100),
                )
                ax.plot(x, y, z, color="green")
                ax.text(cur[0], cur[1], cur[2], f"{row-21}", color="green")
                prev = cur

    #     scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    #     ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    world_limits = ax.get_w_lims()
    ax.set_box_aspect(
        (
            world_limits[1] - world_limits[0],
            world_limits[3] - world_limits[2],
            world_limits[5] - world_limits[4],
        )
    )


def plot_3D_hand(pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_pose(ax, pose, plot_obj=False)
    plt.show()


def plot_3D_gt(gt, img_path, gt2d):
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    plot_pose(ax1, gt)
    ax1.set_title("Ground truth")
    ax2 = fig.add_subplot(122)
    img = Image.open(img_path)
    print(f"Image shape: {img.size}")
    w, h = img.size
    if w != 224:
        size, crop_sz = 256, 224
        new_size = size, size
        if w > h:
            new_size = (size, h * size // w)
        new_img = img.resize(new_size)
        width, height = new_img.size  # Get dimensions
        left = (width - crop_sz) // 2
        top = (height - crop_sz) // 2
        right = (width + crop_sz) // 2
        bottom = (height + crop_sz) // 2

        # Crop the center of the image
        img = new_img.crop((left, top, right, bottom))
    imgDraw = ImageDraw.Draw(img)
    faces = [
        [0, 1, 2, 3, 0],
        [0, 1, 5, 4, 0],
        [0, 3, 7, 4, 0],
        [1, 5, 6, 2, 1],
        [2, 3, 7, 6, 2],
        [5, 6, 7, 4, 5],
    ]
    for face in faces:
        prev = gt2d[21 + face[0], :]
        for idx in face:
            row = 21 + idx
            cur = gt2d[row, :]
            cur, prev = cur.flatten(), prev.flatten()
            imgDraw.line((prev[0], prev[1], cur[0], cur[1]), fill="green")
            prev = cur
    ax2.imshow(img)
    ax2.set_title(f"Input image: {img_path}")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.show()


def plot_3D_pred_gt(pred, img, gt=None):
    fig = plt.figure()
    if gt is not None:
        ax1 = fig.add_subplot(131, projection="3d")
        plot_pose(ax1, pred)
        ax1.set_title("Prediction")
        ax2 = fig.add_subplot(132, projection="3d")
        plot_pose(ax2, gt)
        ax2.set_title("Ground truth")
        ax3 = fig.add_subplot(133)
        ax3.imshow(img)
        ax3.set_title("Input image")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
    else:
        ax1 = fig.add_subplot(121, projection="3d")
        plot_pose(ax1, pred)
        ax1.set_title("Prediction")
        ax2 = fig.add_subplot(122)
        ax2.imshow(img)
        ax2.set_title("Input image")
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
    plt.show()


def select_cnn_model(cnn_def: str, hand_only: bool) -> torch.nn.Module:
    cnn_def = cnn_def.lower()
    if cnn_def == "resnet10":
        cnn = ResNet(model="10", pretrained=True, hand_only=hand_only)
    elif cnn_def == "resnet12":
        cnn = ResNet12()
    elif cnn_def == "resnet18":
        cnn = ResNet(model="18", pretrained=True, hand_only=hand_only)
    elif cnn_def == "resnet34":
        cnn = ResNet(model="34", pretrained=True, hand_only=hand_only)
    elif cnn_def == "resnet50":
        cnn = ResNet(model="50", pretrained=True, hand_only=hand_only)
    elif cnn_def == "mobilenetv3-small":
        # TODO: Hand only?
        cnn = MobileNet(model="v3-small", pretrained=True)
    elif cnn_def == "mobilenetv3-large":
        # TODO: Hand only?
        cnn = MobileNet(model="v3-large", pretrained=True)
    else:
        raise ValueError(f"{cnn_def} is not a valid CNN definition!")
    return cnn


def compute_OBB_corners(mesh: trimesh.Trimesh) -> np.ndarray:
    # From https://github.com/mikedh/trimesh/issues/573
    half = mesh.bounding_box_oriented.primitive.extents / 2
    return trimesh.transform_points(
        trimesh.bounds.corners([-half, half]),
        mesh.bounding_box_oriented.primitive.transform,
    )


def compute_curve(distances, thresholds, nb_keypoints):
    auc_all = list()
    pck_curve_all = list()
    norm_factor = torch.trapz(torch.ones_like(thresholds), thresholds)
    for kp in range(nb_keypoints):
        pck_curve = []
        for t in thresholds:
            pck = torch.mean((torch.vstack(distances)[:, kp] <= t).type(torch.float))
            pck_curve.append(pck)

        pck_curve_all.append(pck_curve)
        auc = torch.trapz(torch.Tensor(pck_curve), torch.Tensor(thresholds))
        auc /= norm_factor
        auc_all.append(auc)

    auc_all = torch.Tensor(auc_all).mean().item()
    # mean only over keypoints
    pck_curve_all = torch.Tensor(pck_curve_all).mean(dim=0)
    return auc_all, pck_curve_all


def plot_curve(values, thresholds, file_path: str, type="pck"):
    label, title = "", ""
    if type.lower() == "pck":
        label = "Percentage of Correct Keypoints"
        title = "Percentage of Correct Hand Joints (3D)"
    elif type.lower() == "pcp":
        label = "Percentage of Correct Poses"
        title = "Percentage of Correct Object Corners (3D)"
    plt.plot(thresholds, values)
    plt.xlabel("mm Threshold")
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True, linestyle="dashed")
    plt.savefig(file_path)
    plt.clf()
