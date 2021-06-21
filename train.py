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

from algorithm.wrappers import MAML_HOPETrainer, MAML_ResnetTrainer, MAML_GraphUNetTrainer
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

    k_shots = 15
    dataset = DatasetFactory.make_data_loader(
        dataset_name, args.input_file, args.batch_size, args.test, True, k_shots
    )
    # hope_trainer = HOPETrainer(dataset_name, args.input_file, args.batch_size, use_cuda=args.gpu,
    # gpu_number=args.gpu_number)
    # hope_trainer.train(meta_batch_size=1, iterations=10)
    # resnet_trainer = ResnetTrainer(dataset_name, args.input_file, args.batch_size,
    # use_cuda=args.gpu, gpu_number=args.gpu_number)
    # resnet_trainer.train(meta_batch_size=1, iterations=10)
    graphunet_trainer = MAML_GraphUNetTrainer(
        dataset,
        k_shots,
        use_cuda=args.gpu,
        gpu_number=args.gpu_number,
        test_mode=args.test,
    )
    if args.test:
        graphunet_trainer.test(meta_batch_size=32, fast_lr=1e-6, meta_lr=1e-5)
    else:
        graphunet_trainer.train(
            meta_batch_size=32, iterations=1000, fast_lr=1e-6, meta_lr=1e-5
        )


if __name__ == "__main__":
    args = parse_args_function()
    main(args)
