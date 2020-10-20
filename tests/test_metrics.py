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
"""Test metrics module."""

import numpy as np
import pytest
import spond.metrics


def test_mapping_accuracy():
    """Test mapping accuracy."""
    f_x = np.array([
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
    
    y = np.array([
        [0.35935493, 0.52762284],
        [0.28900772, 0.69764302],
        [0.09705504, 0.85132928],
        [0.14825254, 0.12129033],
        [0.45746927, 0.73672528],
        [0.98917185, 0.09074136],
        [0.98320845, 0.82869022],
        [0.70395342, 0.78155301],
        [0.22525633, 0.77559237],
        [0.22525633, 0.7187863 ]
    ])
    top1_desired = .8
    top5_desired = 1.
    
    top1a = spond.metrics.mapping_accuracy(f_x, y)
    assert len(top1a) == 1
    assert top1a[0] == top1_desired

    top1b = spond.metrics.mapping_accuracy(f_x, y, n=[1])
    assert len(top1b) == 1
    assert top1b[0] == top1_desired

    top5 = spond.metrics.mapping_accuracy(f_x, y, n=[5])
    assert len(top5) == 1
    assert top5[0] == top5_desired

    top_n = spond.metrics.mapping_accuracy(f_x, y, n=[1, 5])
    assert len(top_n) == 2
    assert top_n[0] == top1_desired
    assert top_n[1] == top5_desired


def test_matrix_comparison():
    """Test matrix correlation."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))

    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))

    score = spond.metrics.alignment_score(a, b)
    np.testing.assert_almost_equal(score, 0.942857142857143)
