#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

from util.factory import DatasetFactory, AlgorithmFactory
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import logging
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    for test_objs in range(1, cfg.experiment.hold_out):
        logging.info(f"[*] Testing with {test_objs} objects")
        dataset = DatasetFactory.make_data_loader(
            cfg,
            to_absolute_path(cfg.shapenet_root),
            cfg.experiment.dataset,
            to_absolute_path(cfg.experiment.dataset_path),
            cfg.experiment.batch_size,
            True,
            cfg.experiment.k_shots,
            cfg.experiment.n_queries,
            cfg.hand_only,
            object_as_task=cfg.experiment.object_as_task,
            normalize_keypoints=cfg.experiment.normalize_keypoints,
            augment_fphad=cfg.experiment.augment,
            test_objects=test_objs,
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
            True,
            model_path=to_absolute_path(cfg.experiment.saved_model)
            if cfg.experiment.saved_model
            else None,
            hand_only=cfg.hand_only,
            use_cuda=cfg.use_cuda,
            gpu_numbers=cfg.gpu_numbers,
        )
        trainer.test(
            batch_size=cfg.experiment.batch_size,
            runs=cfg.test_runs,
            fast_lr=cfg.experiment.fast_lr,
            meta_lr=cfg.experiment.meta_lr,
            visualize=cfg.vis,
            plot=cfg.plot_curves,
            test_objects=test_objs,
        )

if __name__ == "__main__":
    main()
