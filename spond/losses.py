
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

import torch
import torch.distributions as D

"""Losses module."""

def _euclidean_distance(x, y):
    """L2 distance."""
    return torch.mean(torch.sum(torch.pow(x - y, 2), 1))


def sup_loss_func(x_sup, f_y_sup, loss_sup_scale):
    """Supervised loss."""
    loss = loss_sup_scale * _euclidean_distance(x_sup, f_y_sup)
    return loss


def cycle_loss_flex(X, gf_x, Y=None, fg_y=None, loss_cycle_scale=1):
    """
    Calculate cycle consistency loss for a system and its mapping back
    to itself through the model (l1 norm of distances between points)

    Args:
     - X: Original system, tensor
     - gf_x: Resulting system for comparison to original. Tensor with 
     same shape as X. Assumes points correspond to those in X
     - Y and gf_y (optional): Second system

    Output:
     - tot_loss: cycle loss per concept
    """

    if Y == None:
        loss = loss_cycle_scale * _euclidean_distance(X, gf_x)

    elif Y is not None and fg_y is not None:

        loss = (loss_cycle_scale * 0.5 
                * _euclidean_distance(X, gf_x) + _euclidean_distance(Y, fg_y)
                )

    return loss


def create_gmm(system, gmm_scale=0.05):

    """
    Generate probability distribution using gaussian kernels on a
    system of points

    Args:
     - system: set of points from which gmm will be produced
     - batches: bool indicating if system shape includes batch dimension
     - kernel_size: stdev of kernel placed on each point to form gmm

    Output: 
     - gmm_x: gmm probability distribution
    """

    system = torch.squeeze(system)
    n_dim = system.shape[-1]
    n_concepts = system.shape[-2]

    # Weight concepts equally
    mix = D.Categorical(torch.ones(n_concepts,))
    
    # Covariance matrix (diagonal) set with gmm_scale
    components = D.Independent(D.Normal(system, gmm_scale * torch.ones(n_dim,)), 1)
    gmm_X = D.mixture_same_family.MixtureSameFamily(mix, components)
    
    return gmm_X


def negloglik(dist, sample, dist_loss_scale):
    """
    Calculate loglikelihood of drawing a sample from a probability
    distribution

    Args:
     - dist: probability distribution (e.g, output of create_gmm)
    """
    result = -torch.mean(dist.log_prob(sample.double()), axis = 0)
    return result
