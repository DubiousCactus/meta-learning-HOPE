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

from algorithm.wrappers import (
    MAML_HOPETrainer,
    MAML_ResnetTrainer,
    MAML_GraphUNetTrainer,
)
from HOPE.utils.options import parse_args_function
from data.factory import DatasetFactory


def main(args):
    # TODO: Parse from args instead
    dataset_name = None
    if "fphad" in args.input_file.lower() or "fhad" in args.input_file.lower():
        dataset_name = "fphad"
    elif "obman" in args.input_file.lower():
        dataset_name = "obman"
    elif (
        "ho3d" in args.input_file.lower()
        or "ho-3d" in args.input_file.lower()
        or "ho_3d" in args.input_file.lower()
    ):
        dataset_name = "ho3d"
    else:
        raise Exception(f"Unrecognized dataset in {args.input_file}")

    k_shots, n_querries = 15, 30
    dataset = DatasetFactory.make_data_loader(
        dataset_name,
        args.input_file,
        args.batch_size,
        args.test,
        k_shots,
        n_querries,
        object_as_task=True,
    )
    # TODO: Add a model part arg and create a factory for the trainer
    graphunet_trainer = MAML_GraphUNetTrainer(
        dataset,
        k_shots,
        n_querries,
        use_cuda=args.gpu,
        gpu_number=args.gpu_number,
        test_mode=args.test,
    )
    if args.test:
        graphunet_trainer.test(meta_batch_size=4, fast_lr=1e-6, meta_lr=1e-5)
    else:
        graphunet_trainer.train(
            meta_batch_size=12, iterations=1000, fast_lr=1e-6, meta_lr=1e-3
        )


if __name__ == "__main__":
    args = parse_args_function()
    main(args)
