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
"""Evaluation metrics.

Functions:
    mapping_accuracy: Compute the top-n mapping accuracy between two
        systems.

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def mapping_accuracy(f_x, y, n=[1]):
    """Compute mapping accuracy between f_x and y.

    Assumes input arguments `f_x` and `y` are aligned.

    Note: The option of passing in a list of top-n values is provided
    so that the nearest neighbors are only computed once.

    Arguments:
        f_x: A 2D NumPy array of embeddings.
            shape=(n_concept, n_dim)
        y: A 2D NumPy array of embeddings.
            shape=(n_concept, n_dim)
        n (optional): A list of integers indicating the top-n accuracy
            values that should be computed. By default, only the top-1
            accuracy will be computed.

    Returns:
        accuracy: A list of accuracy values in the same order as
            requested by argument `n`.

    """
    n_concept = f_x.shape[0]

    # Create nearest neighbor graph for `y`.
    neigh = NearestNeighbors(n_neighbors=np.max(n))
    neigh.fit(y)
    # Determine which concepts of `y` are closest for each point in `f_x`.
    _, indices = neigh.kneighbors(f_x)

    dmy_idx = np.arange(n_concept)
    dmy_idx = np.expand_dims(dmy_idx, axis=1)

    locs = np.equal(indices, dmy_idx)

    accuracy = []
    for top_i in n:
        accuracy.append(np.mean(np.sum(locs[:, 0:top_i], axis=1)))

    return accuracy
