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
"""Predefined dataset module."""
import numpy as np

def load_noisy_gaussian(n_concept, noise=0, n_dim=2, seed=7849):
    """Load synthetic embeddings drawn from a multivariate Gaussian.

    Arguments:
        n_concept: A scalar indicating the number of concepts.
        noise (optional): A scalar indicating the amount of noise to
            add. This should be between 0 and 1.
        n_dim (optional): A scalar indicating the number of dimensions.
        seed (optional): A scalar indicating the seed to use in the
            random number generator.

    Returns:
        z_0: The first embedding.
        z_1: The second embedding.

    """
    # Create synthetic embeddings.
    np.random.seed(seed)
    z_0 = np.random.randn(n_concept, n_dim)
    noise = noise * np.random.randn(n_concept, n_dim)
    z_1 = z_0 + noise
    return z_0, z_1
