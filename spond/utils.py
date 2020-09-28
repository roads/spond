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
"""Miscellaneous utility module."""

import json
import numpy as np
from sklearn.mixture import GaussianMixture

def parse_config(config_path: str) -> dict:
    """
    Parse training config.

    Parse configuration for training from a json file
    """
    with open(config_path) as r:
        config = json.load(r)
    return config


def preprocess_embedding(z):
    """Pre-process embedding."""
    # Normalize coordinate system.
    gmm = GaussianMixture(n_components=1, covariance_type='spherical')
    gmm.fit(z)
    mu = gmm.means_[0]
    z_norm = z - mu
    max_val = np.max(np.abs(z_norm))
    z_norm /= max_val
    z_norm /= 2
    return z_norm
    