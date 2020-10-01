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

import numpy as np
from sklearn.mixture import GaussianMixture
import torch.optim as optim

def calculate_ceiling(z_1p, z_1):

    """Input systems in contiguous order"""
    z_1p = torch.squeeze(z_1p)
    z_1 = torch.squeeze(z_1)
    n_concept = z_1.shape[0]
    map_idx = np.zeros([n_concept], dtype=int)

    for i_concept in range(n_concept):
        z_1p_i = torch.unsqueeze(z_1p[i_concept], dim=0)
        # Determine closest point in Euclidean space.
        d = torch.sum((z_1p_i - z_1)**2, dim=1)**.5
        map_idx[i_concept] = np.argsort(d.numpy())[0]

    correct = range(n_concept)
    compare = np.sum(map_idx == correct)

    return compare/n_concept

   
def get_output(restart, epoch, cycle_loss, dist_loss, acclist, verbose=0):

        if verbose == 0:
            return None
        
        elif verbose == 1:
            template_loss = 'Restart {0}, epoch {1} Loss | cycle: {2:.3g} | dist: {3:.3g}'
            loss_output = template_loss.format(
                                            restart+1, epoch,
                                            cycle_loss,
                                            dist_loss
                                            )
            template_acc = '{0} Accuracy | 1: {1:.2f} | 5: {2:.2f} | 10: {3:.2f} | half: {4:.2f}'
            acc_output = template_acc.format(
                                '\nMean accuracy', 
                                acclist[0], 
                                acclist[1], 
                                acclist[2], 
                                acclist[3])
            return [loss_output, acc_output]


def pairwise_distance(systemA, systemB):
    """
    Calculate pairwise distances between points in two systems

    Args:
    - systemA and systemB: nxd
    """

    n = systemA.shape[0]    
    B_transpose = np.transpose(systemB)
        
    inner = -2 * np.matmul(systemA, B_transpose)
    
    A_squares = np.sum(
        np.square(systemA), axis=-1
        )
    A_squares = np.transpose(np.tile(A_squares, (n,1)))
        
    B_squares = np.transpose(
        np.sum(np.square(systemB), axis=-1)
        )   
    B_squares = np.tile(B_squares, (n, 1))

    pairwise_distances = np.sqrt(
        np.abs(
            inner + A_squares + B_squares
            )
    )
    
    return pairwise_distances


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


def _get_optimizer(params, optimizer_type="Adam", lr=0.1):
        """
        Create optimizer for training of type specified by input string
        """
        if optimizer_type == "Adam":
            return optim.Adam(params, lr)
        
        elif optimizer_type == "SGD":
            return optim.SGD(params, lr)
