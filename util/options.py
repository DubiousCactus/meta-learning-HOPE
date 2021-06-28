#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Argument parsing.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--dataset", default="./datasets/obman/", help="Input directory", required=True
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/model-",
        help="Prefix of output pkl filename",
        required=True,
    )
    # Optional arguments.
    parser.add_argument("--train", action="store_true", help="Training mode.")
    parser.add_argument("--val", action="store_true", help="Use validation set.")
    parser.add_argument("--test", action="store_true", help="Test model.")
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=6,
        help="Meta-Batch size (number of tasks per meta-training iteration).",
    )
    parser.add_argument(
        "--model-def",
        default="HopeNet",
        help="Name of the model 'HopeNet', 'ResNet', 'GraphUNet' or 'GraphNet'",
        choices=["HopeNet", "ResNet", "GraphUNet", "GraphNet"],
    )
    parser.add_argument("--load-ckpt", type=str, help="Load trained model file")
    parser.add_argument(
        "--gpu-number",
        type=int,
        nargs="+",
        default=[0],
        help="Identifies the GPU number to use.",
    )
    parser.add_argument(
        "--fast-lr",
        type=float,
        default=0.0001,
        help="Learners's learning rate.",
    )
    parser.add_argument(
        "--meta-lr",
        type=float,
        default=0.001,
        help="Meta-Learners's learning rate.",
    )
    parser.add_argument(
        "--val-epoch", type=int, default=1, help="Run validation on epochs."
    )
    parser.add_argument(
        "--num-iterations", type=int, default=20000, help="Maximum number of epochs."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="maml",
        help="Meta-training algorithm.",
        choices=["maml", "fomaml"],
    )
    parser.add_argument(
        "--k-shots", type=int, default=15, help="K-Shot value for Meta-Training."
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=30,
        help="Number of query/target samples for Meta-Training.",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=1,
        help="Number of inner-loop trianing steps for Meta-Training.",
    )
    args = parser.parse_args()
    return args
