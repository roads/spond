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

from .constants import DEVICE

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
    mixture = D.Categorical(torch.ones(n_concept,).to(DEVICE))
    # Diagonal covariance matrix scaled by gmm_scale
    components = D.Independent(D.Normal(fx, gmm_scale * torch.ones(n_dim,).to(DEVICE)), 1)
    gmm_fx = D.mixture_same_family.MixtureSameFamily(mixture, components)

    # Negative log-likelihood
    neg_ll = - torch.mean(gmm_fx.log_prob(gmm_y_samples).to(DEVICE), axis=0)
    return neg_ll
