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
"""Test utils module."""

import numpy as np
import pytest
import spond.utils


def test_preprocess_embedding():
    """Test preprocess embedding."""
    z = np.array([
        [0.35935493, 0.52762284],
        [0.28900772, 0.69764302],
        [0.09705504, 0.85132928],
        [0.14825254, 0.12129033],
        [0.45746927, 0.73672528],
        [0.98917185, 0.09074136],
        [0.98320845, 0.82869022],
        [0.70395342, 0.78155301],
        [0.22525633, 0.7187863 ],
        [0.22525633, 0.77559237]
    ])

    z_p = spond.utils.preprocess_embedding(z)
    z_p_mu = np.mean(z_p, axis=0)
    np.testing.assert_almost_equal(z_p_mu, np.zeros([2]))

    max_abs_val = np.max(np.abs(z_p))
    assert max_abs_val <= .5
