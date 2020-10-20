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
    mapping_accuracy:
    alignment_score: Compute the alignment score betweeen two sets of
        points.
    pdist_triu: Compute the upper-triangular pairwise distances
        between a set of points.

"""

import numpy as np
from scipy.stats import spearmanr
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


def alignment_score(x, y, f=None):
    """Compute the alignment score between two set of points.

    Arguments:
        x: A set of points.
            shape=(n,d)
        y: A set of points.
            shape=(n,d)
        f (optional): A kernel function that computes the similarity
            or dissimilarity between two vectors. The function must
            accept two matrices with shape=(m,d).
    
    Returns:
        corr: The alignment score between the two sets of points.

    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(
            "The argument `x` and `y` must have the same number of rows."
        )

    # Determine upper triangular pairwise distance.
    d_x = pdist_triu(x)
    d_y = pdist_triu(y)

    corr, pval = spearmanr(d_x, d_y)
    return corr


def pdist_triu(x, f=None):
    """Pairwise distance.
    
    Arguments:
        x: A set of points.
            shape=(n,d)
        f (optional): A kernel function that computes the similarity
            or dissimilarity between two vectors. The function must
            accept two matrices with shape=(m,d).
    
    Returns:
        Upper triangular pairwise distances in "unrolled" form.

    """
    n = x.shape[0]
    
    if f is None:
        # Use Euclidean distance.
        def f(x, y):
            return np.sqrt(np.sum((x - y)**2, axis=1))

    # Determine indices of upper triangular matrix (not including
    # diagonal elements).
    idx_upper = np.triu_indices(n, 1)

    return f(x[idx_upper[0]], x[idx_upper[1]])