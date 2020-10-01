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
"Datasets module"

import numpy as np
import torch

import scipy as sp
from scipy import stats

# Generate synthetic embeddings for n_systems
def create_n_systems(n_systems=1,
                    n_epicentres=1, 
                    epicentre_range=1,  
                    n_dim=2, 
                    num_concepts=200, 
                    sigma=1, 
                    noise_size=0.1,  
                    return_noisy = False, 
                    rotation = True):


    """
    Creates n_systems 'clumpy' systems of points. Returns a list of 
    length n_systems, where list[i] is the tensor for system i. 
    
    Args:
        - n_epicentres: number of gaussian 'clumps' to draw from
        - epicentre_range: 1/2 width of uniform dist from which 
        each coordinate of each epicentre's mean is drawn
        - n_dim: dimensionality of distributions and resulting data
        - num_concepts: number of concepts in each resulting system
        - sigma: variance of each gaussian 'clump'
        - noise_size: size of kernel for ranfom noise added to each 
        point in transformation to another system
        - n_systems: number of systems to create
        -return_noisy: if set to True, function returns a second 
        list of length n containing the unrotated versions of the systems

    """

    # Create first system, X
    X_cov = np.zeros((n_dim, n_dim), float)
    np.fill_diagonal(X_cov, sigma)
    
    # Randomly sample epicentre means from specified range
    means = np.random.uniform(
        -epicentre_range, epicentre_range, size = (n_epicentres, n_dim)
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
        torch.tensor(X, dtype = torch.double),0
        )

    systems = [X_tensor]
    return_noisy_X = [X_tensor]

    # For each in number of specified systems
    for i in range(n_systems-1):
        # Generate random rotation matrix
        random_rot_mat = sp.stats.special_ortho_group.rvs(n_dim)
    
        # Generate noisy X
        noisy_X = (X + np.random.multivariate_normal(
                            mean = [0]*n_dim, 
                            cov = X_cov * noise_size,
                            size = num_concepts
                            ))

        # If returning noisy X in separate list, add to list
        if return_noisy == True:
            noisy_X_tensor = torch.unsqueeze(
                torch.tensor(noisy_X, dtype = torch.double), 0
                )
            return_noisy_X.append(noisy_X_tensor)
    
        if rotation == True:
        # Create Y by rotating noisy X
            Y = np.matmul(random_rot_mat, noisy_X.T)
            Y = Y.T
        else:
            Y = noisy_X

        Y = torch.tensor(Y, dtype = torch.double)
        Y = torch.unsqueeze(Y, 0)
        systems.append(Y)

    if return_noisy == False:
        return systems
    
    elif return_noisy == True:
        return systems, return_noisy_X


