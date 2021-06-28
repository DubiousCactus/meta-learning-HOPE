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
from util.options import parse_args

import os


def main(args):
    # TODO: Parse from args instead
    dataset_name = None
    if "fphad" in args.dataset.lower() or "fhad" in args.dataset.lower():
        dataset_name = "fphad"
    elif "obman" in args.dataset.lower():
        dataset_name = "obman"
    elif (
        "ho3d" in args.dataset.lower()
        or "ho-3d" in args.dataset.lower()
        or "ho_3d" in args.dataset.lower()
    ):
        dataset_name = "ho3d"
    else:
        raise Exception(f"Unrecognized dataset in {args.dataset}")

    if not os.path.isfile('./config.yaml'):
        raise Exception("Config file missing! Read README.md.")
    dataset = DatasetFactory.make_data_loader(
        dataset_name,
        args.dataset,
        args.meta_batch_size,
        args.test,
        args.k_shots,
        args.n_queries,
        object_as_task=True,
    )
    trainer = AlgorithmFactory.make_training_algorithm(
        args.algorithm,
        args.model_def,
        dataset,
        os.path.join("./checkpoints", args.checkpoint_name),
        args.k_shots,
        args.n_queries,
        args.inner_steps,
        model_path=args.load_ckpt,
        test_mode=args.test,
        use_cuda=True,
        gpu_number=args.gpu_number,
    )
    if args.test:
        trainer.test(
            meta_batch_size=args.meta_batch_size,
            fast_lr=args.fast_lr,
            meta_lr=args.meta_lr,
        )
    else:
        trainer.train(
            meta_batch_size=args.meta_batch_size,
            iterations=args.num_iterations,
            fast_lr=args.fast_lr,
            meta_lr=args.meta_lr,
            save_every=args.save_every,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
