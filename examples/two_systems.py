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
"""Example that uses 2D synthetic data.

Can optionally use semi-supervision.

"""
import argparse
import scipy.stats
import numpy as np

from spond.models import Aligner
from spond.datasets import load_noisy_gaussian
from spond.metrics import mapping_accuracy
from spond.utils import parse_config, preprocess_embedding


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Unsupervised alignment of conceptual systems'
    )
    parser.add_argument(
        '--log', help='log to record learning curve', default=None
    )

    args = parser.parse_args()

    # Experimental 2D sythetic Data
    n_concept = 200
    noise = 0.01
    n_dim = 2

    z_0, z_1 = load_noisy_gaussian(
        n_concept=n_concept, noise=noise, n_dim=n_dim
    )
    
    z_0 = preprocess_embedding(z_0)
    z_1 = preprocess_embedding(z_1)

    template = 'Ceiling Accuracy 1: {0:.2f} 5: {1:.2f} 10: {2:.2f}\n'
    acc_1, acc_5, acc_10 = mapping_accuracy(
        z_0, z_1, n=[1, 5, 10]
    )
    print(template.format(acc_1, acc_5, acc_10))
    
    # Add random rotation to the second embedding.
    np.random.seed(42)
    rot_mat = scipy.stats.special_ortho_group.rvs(z_0.shape[1])
    z_1 = np.matmul(z_1, rot_mat)

    # Shuffle second embedding, but keep track of correct mapping.
    n_concept = z_0.shape[0]
    idx_rand = np.random.permutation(n_concept)
    z_1_shuffle = z_1[idx_rand, :]
    # Determine correct mapping.
    y_idx_map = np.argsort(idx_rand)
    # Verify mapping to be safe.
    np.testing.assert_array_equal(z_1, z_1_shuffle[y_idx_map, :])
    
    # Semi-supervision
    use_semisupervision = False
    n_supervised = 10

    if use_semisupervision:
        sup_idx_x = np.random.choice([i for i in range(n_concept)], n_supervised)
        sup_idx_y = y_idx_map[sup_idx_x]
    else:
        sup_idx_x = None
        sup_idx_y = None


    # Alignment
    aligner = Aligner(
        x=z_0,
        y=z_1_shuffle,
        y_idx_map=y_idx_map,
        sup_idx_x=sup_idx_x,
        sup_idx_y=sup_idx_y
    )

    aligner.train(
        n_restart=100,
        max_epoch=30,
        hidden_size=100,
        gmm_scale=0.01,
        loss_set_scale=1.0,
        loss_cycle_scale=2.0,
        loss_sup_scale=2.0
    )
    