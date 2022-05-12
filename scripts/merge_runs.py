#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Read in a csv files of measurements, merge them all into one CSV file, compute mean into other CSV
file.
"""

import numpy as np
import sys
import csv


def parse_file(path: str):
    header = None
    val = []
    indices = []
    with open(path, "r") as file:
        csv_obj = csv.reader(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
        for line in csv_obj:
            if header is None:
                header = line
                continue
            if line != []:
                val.append(float(line[1]))
                indices.append(int(line[0]))
    return header, indices, val

def write_mean_file(path: str, header, indices, val):
    with open(path, "w") as file:
        csv_obj = csv.writer(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
        val = np.mean(val, axis=0)
        print(f"[*] Variance: {np.var(val):.4f}")
        print(f"[*] Standard deviation: {np.std(val):.4f}")
        csv_obj.writerow(header)
        for i in range(val.shape[0]):
            csv_obj.writerow([indices[0,i], val[i]])

def write_merged_file(path: str, header, indices, val):
    with open(path, "w") as file:
        csv_obj = csv.writer(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
        csv_obj.writerow(header)
        for j in range(val.shape[1]):
            for i in range(val.shape[0]):
                csv_obj.writerow([indices[i,j], val[i,j]])

    # Also write an origin-aligned file
    or_al_path = path.split('.')[0] + "_origin.csv"
    with open(or_al_path, "w") as file:
        csv_obj = csv.writer(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
        csv_obj.writerow(header)
        for j in range(val.shape[1]): # For all samples
            for i in range(val.shape[0]): # In batch i
                csv_obj.writerow([indices[i,j], val[i,j]-val[i, 0]])

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print(f"Usage: {sys.argv[0]} <file_1> <file_2> ... <file_n> <merged_file_name> <mean_file_name>")
        exit(1)
    n_files = len(sys.argv) - 3
    merged_file_name, mean_file_name = sys.argv[-2], sys.argv[-1]
    prev_indices = []
    mega_val, mega_indices = [], []
    header = None
    # idx, values = [], []

    for i in range(n_files):
        h, indices, values = parse_file(sys.argv[i+1])
        if prev_indices != []:
            assert len(prev_indices) == len(indices), "Length of both files must match"
        else:
            header = h
        print(values)
        mega_val.append(values)
        mega_indices.append(indices)
        prev_indices = indices
    indices = np.vstack(mega_indices)
    values = np.vstack(mega_val)
    write_mean_file(mean_file_name, header, indices, values)
    write_merged_file(merged_file_name, header, indices, values)
