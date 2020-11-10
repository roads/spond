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
import scipy as sp
import torch


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


def create_n_systems(n_systems=1, n_epicentres=1, epicentre_range=1,  n_dim=2,
                     num_concepts=200, sigma=1, noise_size=0.1,
                     return_noisy=False, rotation=True):
    """Create n_systems 'clumpy' systems of points.

    Arguments:
        n_epicentres: number of gaussian 'clumps' to draw from
        epicentre_range: 1/2 width of uniform dist from which
        each coordinate of each epicentre's mean is drawn
        n_dim: dimensionality of distributions and resulting data
        num_concepts: number of concepts in each resulting system
        sigma: variance of each gaussian 'clump'
        noise_size: size of kernel for ranfom noise added to each
        point in transformation to another system
        n_systems: number of systems to create
        return_noisy: if set to True, function returns a second
        list of length n containing the unrotated versions of the systems

    Returns:
        systems: list of embeddings
        noisy_systems (if return_noisy is True): list of embeddings with
        only noise added, no rotation
    """
    # Create first system, X
    X_cov = np.zeros((n_dim, n_dim), float)
    np.fill_diagonal(X_cov, sigma)

    # Randomly sample epicentre means from specified range
    means = np.random.uniform(
        -epicentre_range, epicentre_range, size=(n_epicentres, n_dim)
        )

    X = []
    for i in range(num_concepts):

        # Assign concept to an epicentre
        mean = i % n_epicentres

        # Take sample from relevant epicentre
        value = np.random.multivariate_normal(
            mean=means[mean], cov=X_cov,  size=1
            )
        # Append to list of points in system
        X.append(value)
    X = np.squeeze(X)
    X = np.array(X)

    # Add tensor to output list of systems
    X_tensor = torch.unsqueeze(
        torch.tensor(X, dtype=torch.double), 0
        )

    systems = [X_tensor]
    return_noisy_X = [X_tensor]

    # For each in number of specified systems
    for i in range(n_systems-1):
        # Generate random rotation matrix
        random_rot_mat = sp.stats.special_ortho_group.rvs(n_dim)

        # Generate noisy X
        noisy_X = (X + np.random.multivariate_normal(
                            mean=[0]*n_dim,
                            cov=X_cov * noise_size,
                            size=num_concepts
                            ))

        # If returning noisy X in separate list, add to list
        if return_noisy is True:
            noisy_X_tensor = torch.unsqueeze(
                torch.tensor(noisy_X, dtype=torch.double), 0
                )
            return_noisy_X.append(noisy_X_tensor)

        if rotation is True:
            # Create Y by rotating noisy X
            Y = np.matmul(random_rot_mat, noisy_X.T)
            Y = Y.T
        else:
            Y = noisy_X

        Y = torch.tensor(Y, dtype=torch.double)
        Y = torch.unsqueeze(Y, 0)
        systems.append(Y)

    if return_noisy is False:
        return systems

    elif return_noisy is True:
        return systems, return_noisy_X
