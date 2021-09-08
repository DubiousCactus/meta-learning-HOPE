#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
import torch.nn


class InitWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.randomly_initialize_weights = True
