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
"""Evaluation metrics."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# TODO: refactor
def mapping_accuracy(f_x, y):
    """Compute mapping accuracy.

    Assumes inputs f_x and y are aligned.

    """
    n_concept = f_x.shape[0]
    n_half = int(np.ceil(n_concept / 2))

    # Create nearest neighbor graph for y.
    neigh = NearestNeighbors(n_neighbors=n_half)
    neigh.fit(y)
    # Determine which concepts of y are closest for each point in f_x.
    _, indices = neigh.kneighbors(f_x)

    dmy_idx = np.arange(n_concept)
    dmy_idx = np.expand_dims(dmy_idx, axis=1)

    locs = np.equal(indices, dmy_idx)

    is_correct_half = np.sum(locs[:, 0:n_half], axis=1)
    is_correct_10 = np.sum(locs[:, 0:10], axis=1)
    is_correct_5 = np.sum(locs[:, 0:5], axis=1)
    is_correct_1 = locs[:, 0]

    acc_half = np.mean(is_correct_half)
    acc_10 = np.mean(is_correct_10)
    acc_5 = np.mean(is_correct_5)
    acc_1 = np.mean(is_correct_1)
    return acc_1, acc_5, acc_10, acc_half