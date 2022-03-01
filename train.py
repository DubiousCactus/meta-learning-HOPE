#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

from util.factory import DatasetFactory, AlgorithmFactory
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
        False,
        cfg.experiment.k_shots,
        cfg.experiment.n_queries,
        cfg.hand_only,
        object_as_task=cfg.experiment.object_as_task,
        normalize_keypoints=cfg.experiment.normalize_keypoints,
        augment_fphad=cfg.experiment.augment,
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
        False,
        model_path=to_absolute_path(cfg.experiment.saved_model)
        if cfg.experiment.saved_model
        else None,
        hand_only=cfg.hand_only,
        use_cuda=cfg.use_cuda,
        gpu_numbers=cfg.gpu_numbers,
    )
    table = wandb.Table(
        columns=[
            "Algorithm",
            "Optimizer",
            "Dataset",
            "Batch size",
            "Model",
        ]
    )
    table.add_data(
        cfg.experiment.algorithm,
        cfg.experiment.optimizer,
        cfg.experiment.dataset,
        cfg.experiment.batch_size,
        cfg.experiment.model_def,
    )
    wandb.log({"Config summary": table}, step=0)
    trainer.train(
        batch_size=cfg.experiment.batch_size,
        iterations=cfg.experiment.iterations,
        fast_lr=cfg.experiment.fast_lr,
        meta_lr=cfg.experiment.meta_lr,
        optimizer=cfg.experiment.optimizer.lower(),
        val_every=cfg.experiment.val_every,
        resume=cfg.resume_training,
        use_scheduler=cfg.use_scheduler,
    )


if __name__ == "__main__":
    main()
