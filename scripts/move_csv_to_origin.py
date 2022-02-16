#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Read in a csv file of measurements, substract them all form the first one, save as new csv file.
"""

import numpy as np
import sys
import csv


assert len(sys.argv) == 2
file_name = sys.argv[1]
header = None
val = []
indices = []
with open(file_name, "r") as file:
    csv_obj = csv.reader(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
    for line in csv_obj:
        if header is None:
            header = line
            continue
        if line != []:
            print(line)
            val.append(float(line[1]))
            indices.append(int(line[0]))
with open(file_name.split('.')[0] + "_origin.csv", "w") as file:
    csv_obj = csv.writer(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
    val = np.array(val, dtype=np.float32)
    val -= val[0]
    print(f"[*] Variance: {np.var(val):.4f}")
    print(f"[*] Standard deviation: {np.std(val):.4f}")
    csv_obj.writerow(header)
    for i in range(val.shape[0]):
        print([indices[i], val[i]])
        csv_obj.writerow([indices[i], val[i]])
