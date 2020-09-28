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
"""Runner. Demo with 2D Synthetic Data with/without semi-supervision."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import scipy.stats
import numpy as np

from spond.models import Aligner
from spond.datasets import load_noisy_gaussian
from spond.metrics import mapping_accuracy
from spond.utils import parse_config, preprocess_embedding


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Unsupervised alignment of conceptual systems')
    parser.add_argument('--config_path', help='location for configuration file', default='.config.json')
    parser.add_argument('--log', help='log to record learning curve', default=None)

    args = parser.parse_args()
    config = parse_config('tests/config.json')

    # Experimental 2D sythetic Data
    z_0, z_1 = load_noisy_gaussian(n_concept=200, noise=0.01, n_dim=2)
    
    z_0 = preprocess_embedding(z_0)
    z_1 = preprocess_embedding(z_1)

    template = 'Ceiling Accuracy 1: {0:.2f} 5: {1:.2f} 10: {2:.2f} Half: {3:.2f}\n'
    acc_1, acc_5, acc_10, acc_half = mapping_accuracy(z_0, z_1)
    print(template.format(acc_1, acc_5, acc_10, acc_half))
    
    # Add random rotation to the second embedding.
    np.random.seed(3123)
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

    if use_semisupervision:
        sup_idx_x = np.random.choice([i for i in range(200)], 10)
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
        n_restart=config['n_restart'],
        max_epoch=config['max_epoch'],
        hidden_size=config['hidden_size'],
        gmm_scale=config['gmm_scale'],
        loss_set_scale=config['loss_set_scale'],
        loss_cycle_scale=config['loss_cycle_scale'],
        loss_sup_scale=config['loss_sup_scale']
    )