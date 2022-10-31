#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Read in a list of csv files and merge with averaging the measurements into output file.
"""

import numpy as np
import sys
import csv
import os

assert len(sys.argv) >= 4
header = None
values = {}

for file_name in sys.argv[1:]:
    if not os.path.isfile(file_name):
        print(f"[*] Outputing {file_name}")
        break
    with open(file_name, "r") as file:
        csv_obj = csv.reader(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
        f_header = None
        for line in csv_obj:
            if f_header is None:
                header = line
                f_header = line
                continue
            if line != []:
                print(line)
                idx = int(line[0])
                if idx not in values:
                    values[idx] = [float(line[1])]
                else:
                    values[idx].append(float(line[1]))

with open(sys.argv[-1], "w") as file:
    csv_obj = csv.writer(file, dialect=csv.unix_dialect, quoting=csv.QUOTE_NONE)
    csv_obj.writerow(header)
    for idx, val in values.items():
        csv_obj.writerow([idx, np.array(val).mean()])
