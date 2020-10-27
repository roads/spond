# -*- coding: utf-8 -*-
# Copyright 2020 The Spond Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility module."""

import json

import numpy as np


def preprocess_embedding(z):
    """Pre-process embedding.
    
    Center and scale embedding to be between -.5, and .5.

    Arguments:
        z: A 2D NumPy array.
            shape=(n_concept, n_dim)
    
    Returns:
        z_p: A pre-processed embedding.

    """
    # Center embedding.
    z_p = z - np.mean(z, axis=0, keepdims=True)

    # Scale embedding.
    max_val = np.max(np.abs(z_p))
    z_p /= max_val
    z_p /= 2
    
    return z_p
