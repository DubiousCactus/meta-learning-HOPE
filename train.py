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

from algorithm.wrappers import HOPETrainer, ResnetTrainer, GraphUNetTrainer
from HOPE.utils.options import parse_args_function

# TODO:
# [x] Implement the right data loader such that one task = one object (several sequences per
# object!)
# [x] Implement MAML learning for the entire HOPENet
# [x] Implement MAML learning for the feature extractor only (ResNet)
# [x] Implement MAML learning for Graph U-Net only
# [ ] Implement normal learning for each part and whole HOPENet (just as in the paper)


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

    # hope_trainer = HOPETrainer(dataset_name, args.input_file, args.batch_size, use_cuda=args.gpu,
    # gpu_number=args.gpu_number)
    # hope_trainer.train(meta_batch_size=1, iterations=10)
    # resnet_trainer = ResnetTrainer(dataset_name, args.input_file, args.batch_size,
    # use_cuda=args.gpu, gpu_number=args.gpu_number)
    # resnet_trainer.train(meta_batch_size=1, iterations=10)
    graphunet_trainer = GraphUNetTrainer(
        dataset_name,
        args.input_file,
        args.batch_size,
        5,
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
