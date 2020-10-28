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
"""Dataset loading module."""

import torch
import torch.distributions as D


def _euclidean_distance(x, y):
    """L2 distance."""
    return torch.mean(torch.sum(torch.pow(x - y, 2), 1))


def sup_loss_func(x_sup, f_y_sup, loss_sup_scale):
    """Supervised loss."""
    loss = loss_sup_scale * _euclidean_distance(x_sup, f_y_sup)
    return loss


def cycle_loss_func(x, y, x_c, y_c, loss_cycle_scale):
    """Cycle loss."""
    loss = loss_cycle_scale * 0.5 * (_euclidean_distance(x, x_c) + _euclidean_distance(y, y_c))
    return loss


def set_loss_func(fx, gmm_y_samples, gmm_scale):
    """Statistical distance between two sets.

    Use upperbound of GMM KL divergence approximation.
    Assumes that incoming `fx` and `y` is all of the data.
    """
    # TODO: make the distribution params learnable
    n_concept = fx.shape[0]
    n_dim = fx.shape[1]

    # Equal weight for each component
    mixture = D.Categorical(torch.ones(n_concept,))
    # Diagonal covariance matrix scaled by gmm_scale
    components = D.Independent(D.Normal(fx, gmm_scale * torch.ones(n_dim,)), 1)
    gmm_fx = D.mixture_same_family.MixtureSameFamily(mixture, components)

    # Negative log-likelihood
    neg_ll = - torch.mean(gmm_fx.log_prob(gmm_y_samples), axis=0)
    return neg_ll


def cycle_loss_flex(X, gf_x, Y=None, fg_y=None, loss_cycle_scale=1):
    """
    Calculate cycle consistency loss for a system and its mapping back
    to itself through the model (l1 norm of distances between points)

    Args:
        X: Original system, tensor
        gf_x: Resulting system for comparison to original. Tensor with 
        same shape as X. Assumes points correspond to those in X
        Y and gf_y (optional): Second system

    Returns:
        tot_loss: cycle loss per concept
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

    Arguments:
        system: set of points from which gmm will be produced
        batches: bool indicating if system shape includes batch dimension
        kernel_size: stdev of kernel placed on each point to form gmm

    Returns: 
        gmm_x: gmm probability distribution
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


def negloglik(dist, sample, dist_loss_scale=1):
    """
    Calculate loglikelihood of drawing a sample from a probability
    distribution

    Arguments:
        dist: probability distribution (e.g, output of create_gmm)
        sample: sample for which the negloglik is being calculated
        dist_loss_scale: scaling factor for distribution loss
    """
    result = -torch.mean(dist.log_prob(sample.double()), axis = 0)
    return result * dist_loss_scale















