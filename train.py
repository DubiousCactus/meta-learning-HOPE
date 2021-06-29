#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-Train HOPE-Net or its individual parts.
"""

from data.factory import DatasetFactory, AlgorithmFactory
from omegaconf import DictConfig, OmegaConf

import hydra
import os


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = DatasetFactory.make_data_loader(
        cfg.shapenet_root,
        cfg.experiment.dataset,
        cfg.experiment.dataset_path,
        cfg.experiment.meta_batch_size,
        cfg.test_mode,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        object_as_task=True,
    )
    trainer = AlgorithmFactory.make_training_algorithm(
        cfg.experiment.algorithm,
        cfg.experiment.model_def,
        dataset,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        cfg.experiment.steps,
        cfg.experiment.checkpoint_path,
        model_path=cfg.experiment.saved_model
        if cfg.experiment.saved_model != ""
        else None,
        test_mode=cfg.test_mode,
        use_cuda=cfg.use_cuda,
        gpu_number=cfg.gpu_number,
    )
    if cfg.test_mode:
        trainer.test(
            meta_batch_size=cfg.experiment.meta_batch_size,
            fast_lr=cfg.experiment.fast_lr,
            meta_lr=cfg.experiment.meta_lr,
        )
    else:
        trainer.train(
            meta_batch_size=cfg.experiment.meta_batch_size,
            iterations=cfg.experiment.iterations,
            fast_lr=cfg.experiment.fast_lr,
            meta_lr=cfg.experiment.meta_lr,
            lr_step=cfg.experiment.lr_step,
            lr_step_gamma=cfg.experiment.lr_step_gamma,
            save_every=cfg.save_every,
            resume=cfg.resume_training,
        )


if __name__ == "__main__":
    main()
