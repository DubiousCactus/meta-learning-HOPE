#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Analyse gradients for Table 2.
"""

from util.factory import DatasetFactory, AlgorithmFactory
from algorithm.wrappers.anil import ANIL_CNNTrainer
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import wandb
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="HOPE-Net", config=cfg)

    dataset = DatasetFactory.make_data_loader(
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
    trainer = AlgorithmFactory.make_training_algorithm(
        cfg,
        cfg.experiment.algorithm,
        cfg.experiment.model_def,
        cfg.experiment.cnn_def,
        dataset,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        cfg.experiment.steps,
        cfg.experiment.checkpoint_path,
        cfg.test_mode,
        model_path=to_absolute_path(cfg.experiment.saved_model)
        if cfg.experiment.saved_model
        else None,
        hand_only=cfg.hand_only,
        use_cuda=cfg.use_cuda,
        gpu_numbers=cfg.gpu_numbers,
    )
    assert type(trainer) is ANIL_CNNTrainer, "Can only analyse ANIL"
    trainer.analyse_inner_gradients(
        dataset, cfg.experiment.fast_lr, n_tasks=cfg.analyse_tasks
    )


if __name__ == "__main__":
    main()
