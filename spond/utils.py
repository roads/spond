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

import json

import numpy as np
import torch.optim as optim
import torch


def preprocess_embedding(z):
    """Pre-process embedding.
    
    Center and scale embedding to be between -.5, and .5.

    Arguments:
        z: A 2D NumPy array.
            shape=(n_concept, n_dim)
    
    Returns:
        z_p: A pre-processed embedding.

    """
    # Center embedding.
    z_p = z - np.mean(z, axis=0, keepdims=True)

    # Scale embedding.
    max_val = np.max(np.abs(z_p))
    z_p /= max_val
    z_p /= 2
    
    return z_p


def calculate_ceiling(z_1p, z_1):

    """
    Calculate ceiling accuracy for two systems inputted in 
    contiguous order
    """
    
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

   
def get_output(restart, epoch, batch_loss, set_loss, acclist, verbose=0):
    
    """
    Get output template and format with current loss
    """
        
    loss_list = []
    for loss in [batch_loss, set_loss]:
        l = [np.round(x, 3) for x in loss]
        loss_list.append(l)

    loss_list = [y for x in loss_list for y in x]

    double_form = "disc: {}, gen: {}"
    if len(batch_loss) > 1: 
        batch_loss = double_form.format(loss_list[0], loss_list[1])
    else:
        batch_loss = loss_list[0] 

    if len(set_loss) > 1:
        set_loss = double_form.format(loss_list[-2], loss_list[-1])
    else:
        set_loss = loss_list[-1]

    if verbose == 0:
        return None
    
    elif verbose == 1:
        template_loss = 'Restart {0}, epoch {1}: Loss | batch: {2} | set: {3}'
        loss_output = template_loss.format(
                                        restart+1, epoch,
                                        batch_loss,
                                        set_loss
                                        )
        template_acc = '{0} Accuracy | 1: {1:.2f} | 5: {2:.2f} | 10: {3:.2f}'
        acc_output = template_acc.format(
                            '\nMean', 
                            acclist[0], 
                            acclist[1], 
                            acclist[2])
        return [loss_output, acc_output]
    
    
def _get_optimizer(params, optimizer_type="Adam", lr=0.1):
    """
    Create optimizer for training of type specified by input string
    """
    if optimizer_type == "Adam":
        return optim.Adam(params, lr)

    elif optimizer_type == "SGD":
        return optim.SGD(params, lr)
  
