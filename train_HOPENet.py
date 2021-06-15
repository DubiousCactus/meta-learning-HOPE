#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-Train HOPE-Net
"""

from HOPE.utils.options import parse_args_function
from meta_trainers import HOPETrainer

# TODO:
# [ ] Implement the right data loader such that one task = one object (several sequences per
# object!)
# [x] Implement MAML learning for the entire HOPENet
# [ ] Implement MAML learning for the feature extractor only (ResNet)
# [ ] Implement MAML learning for Graph U-Net only

def main(args):
    # TODO: Parse from args instead
    dataset_name = None
    if "fphad" in args.input_file.lower() or "fhad" in args.input_file.lower():
        dataset_name = "fphad"
    elif "obman" in args.input_file.lower():
        dataset_name = "obman"
    elif ("ho3d" in args.input_file.lower() or "ho-3d" in args.input_file.lower() or "ho_3d" in
            args.input_file.lower()):
        dataset_name = "ho3d"
    else:
        raise Exception(f"Unrecognized dataset in {args.input_file}")

    hope_trainer = HOPETrainer("hopenet", dataset_name, args.input_file, args.learning_rate,
            args.lr_step, args.lr_step_gamma, args.batch_size, use_cuda=args.gpu,
            gpu_number=args.gpu_number)
    hope_trainer.train(1, 10)



if __name__ == "__main__":
    args = parse_args_function()
    main(args)
