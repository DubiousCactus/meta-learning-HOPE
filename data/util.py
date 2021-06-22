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

    return meshes


def load_mesh(model_path: str) -> trimesh.Trimesh:
    """
    Directly copied from: https://github.com/hassony2/obman
    """
    model_path_obj = model_path.replace(".pkl", ".obj")
    if os.path.exists(model_path):
        with open(model_path, "rb") as obj_f:
            mesh = pickle.load(obj_f)
    elif os.path.exists(model_path_obj):
        with open(model_path_obj, "r") as m_f:
            mesh = fast_load_obj(m_f)[0]
    else:
        raise ValueError(
            "Could not find model pkl or obj file at {}".format(
                model_path.split(".")[-2]
            )
        )
    return trimesh.load(mesh)


def compute_obman_labels(
    meta_info: dict, cam_intr: np.ndarray, cam_extr: np.ndarray, shapenet_template: str
) -> tuple:
    # Get the hand coordinates
    hand_coords_2d, hand_coords_3d = (
        torch.Tensor(meta_info["coords_2d"].astype(np.float32)),
        torch.Tensor(
            cam_extr[:3, :3].dot(meta_info["coords_3d"].transpose()).transpose()
        ),
    )
    # 1. Load the mesh (see obman.py)
    obj_path = shapenet_template.format(meta_info["class_id"], meta_info["sample_id"])
    mesh = load_mesh(obj_path)
    # 2. Load the transform
    transform = meta_info["affine_transform"]
    # 3. Obtain the oriented bounding box vertices (x1000?)
    verts = np.array(mesh.bounding_box_oriented.vertices)  # * 1000
    # transformed_mesh = mesh.apply_transform(transform)
    # vertices_3d = transformed_mesh.bounding_box_oriented.vertices
    # 4. Apply the transform to the vertices
    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    trans_verts = transform.dot(hom_verts.T).T[:, :3]
    # 5. Apply the camera extrinsic to the transformed vertices: these are the 3D vertices
    trans_verts = cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
    vertices_3d = np.array(trans_verts).astype(np.float32)
    # 6. Project using camera intrinsics: these are the 2D vertices
    hom_2d_verts = np.dot(cam_intr, vertices_3d.transpose())
    vertices_2d = hom_2d_verts / hom_2d_verts[2, :]
    vertices_2d = vertices_2d[:2, :].transpose()
    return (
        torch.cat([hand_coords_2d, torch.Tensor(vertices_2d)]),
        torch.cat([hand_coords_3d, torch.Tensor(vertices_3d)]),
    )


def mp_process_meta_file(idx_meta, root, cam_intr, cam_extr, shapenet_template):
    '''
    Process a meta-info file for ObMan. Useful for multiprocessing.
    '''
    idx, meta = idx_meta
    obj_id, sample = None, None
    with open(meta, "rb") as meta_file:
        meta_obj = pickle.load(meta_file)
        img_path = os.path.join(root, "rgb", f"{idx}.jpg")
        coord_2d, coord_3d = compute_obman_labels(
            meta_obj, cam_intr, cam_extr, shapenet_template
        )
        obj_id = meta_obj["class_id"]
        sample = (img_path, coord_2d, coord_3d)
    return obj_id, sample
