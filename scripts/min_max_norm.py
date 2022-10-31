#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from io import TextIOWrapper

import argparse


def main(csv: TextIOWrapper):
    print("[*] Loading csv...")
    header = csv.readline()  # Skip the header
    raw_MSEs = [float(line.split(",")[-1].strip()) for line in csv.readlines()]
    _min, _max = min(raw_MSEs), max(raw_MSEs)
    norm_MSEs = [(mse - _min) / (_max - _min) for mse in raw_MSEs]
    print("[*] Dumping to output.csv...")
    with open("output.csv", "w") as output_csv:
        output_csv.write(header)
        output_csv.writelines([f"{i+5},{mse}\n" for i, mse in enumerate(norm_MSEs)])
    print("[*] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to CSV file")
    args = parser.parse_args()
    with open(args.file, "r") as csv:
        main(csv)
