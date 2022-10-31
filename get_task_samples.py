#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
This script was used to generate task samples for the paper.
"""

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.transforms import torch
from tqdm.utils import os
from data.custom import CustomDataset
from data.dataset.base import BaseDatasetTaskLoader
from data.dataset.dex_ycb import DexYCBDatasetTaskLoader
from util.factory import DatasetFactory, AlgorithmFactory
from algorithm.wrappers.anil import ANIL_CNNTrainer
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from PIL import Image

import numpy as np
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    data_loader = DatasetFactory.make_data_loader(
        cfg,
        to_absolute_path(cfg.shapenet_root),
        cfg.experiment.dataset,
        to_absolute_path(cfg.experiment.dataset_path),
        cfg.experiment.batch_size,
        cfg.test_mode,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        cfg.hand_only,
        object_as_task=cfg.experiment.object_as_task,
        normalize_keypoints=cfg.experiment.normalize_keypoints,
        augment_fphad=cfg.experiment.augment,
        auto_load=False,
    )
    assert type(data_loader) is DexYCBDatasetTaskLoader, "Only works with DexYCB"

    samples = data_loader.make_raw_dataset(tiny=cfg.experiment.tiny)
    rand_obj_sequence, obj_sampled = [], []
    keys = list(samples.keys())
    while len(rand_obj_sequence) < 5:
        obj_id, seq_id = keys[np.random.randint(0, len(keys))]
        if obj_id not in obj_sampled:
            rand_obj_sequence.append((obj_id, seq_id))
            obj_sampled.append(obj_id)
    print(
        f"[*] Saving random samples for {', '.join([data_loader.obj_labels[i] for i, _ in rand_obj_sequence])}"
    )
    mean, std = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32), torch.tensor(
        [0.221, 0.224, 0.225], dtype=torch.float32
    )
    unnormalize = transforms.Normalize(
        mean=(-mean / std).tolist(), std=(1.0 / std).tolist()
    )
    base_path = "task_sample_images"
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    print(f"[*] Saving to {os.path.join(os.getcwd(), base_path)}...")
    for obj_id, seq_id in rand_obj_sequence:
        obj_dir = f"obj_{obj_id}"
        if not os.path.isdir(os.path.join(base_path, obj_dir)):
            os.mkdir(os.path.join(base_path, obj_dir))
        # Set object_as_task=False because we pass it a list and not a dict
        task = CustomDataset(
            samples[(obj_id, seq_id)],
            img_transform=BaseDatasetTaskLoader._img_transform,
            object_as_task=False,
        )
        # I'm not using learn2learn because there's only one class so it wouldn't work
        dataset = DataLoader(
            task,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # Disable multiple workers because it'll be faster since we're re-creating dataloaders
        )
        n = 20
        for i, task in enumerate(dataset):
            if i == n:
                break
            images = task[0]
            unnormalized_img = unnormalize(images[0])
            npimg = (
                (unnormalized_img * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )
            img = Image.fromarray(npimg)
            img.save(os.path.join(base_path, obj_dir, f"{i:03d}.jpg"))


if __name__ == "__main__":
    main()
