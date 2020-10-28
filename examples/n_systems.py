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
"""Example that performs N-system alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import datasets
import losses
import models 


## Embedding parameters
n_systems = 4
n_dim = 3
n_concepts = 200
n_epicentres = 5 
epicentre_range = 5 
gaussian_sigma = 0.8 
noise = 0.1 

# Training parameters
restarts = 50
max_epoch = 30
n_batch = 10
cycle_scale = 100 # scale for cycle loss
n_batch = 10 # Batches for cycle loss training each epoch
optimizer = "Adam"
learning_rate = 0.1
gmm_scale = 0.1
restart_schedule = [(0.2, 0.2), # Take best model with p=0.2 first 20% of restarts
                    (0.7, 0.9), # Take best model with p=0.7 from 20-90% of restarts
                    (1, 1)] # Take best model with p=1 after 90% of restarts

# Model parameters
map_type = "Linear" # Currently one of ("Linear", "MLP")
n_hidden = max(n_dim,10)

# Generate synthetic embeddings
systems, noisy_systems = datasets.create_n_systems(
                                            n_systems=n_systems,
                                            n_epicentres=n_epicentres, 
                                            epicentre_range=epicentre_range, 
                                            n_dim=n_dim, 
                                            num_concepts=n_concepts, 
                                            sigma=gaussian_sigma, 
                                            noise_size=noise,  
                                            return_noisy=True, 
                                            rotation = True
                                            )


data_list = []
idx_list = []

for i in range(n_systems):

    sys = systems[i]
    sys = utils.preprocess_embedding(torch.squeeze(sys).detach().numpy())

    # Random shuffle index
    shuff_idx = torch.randperm(n_concepts)

    # Shuffle system
    sys_shuff = torch.Tensor(sys[shuff_idx, :])

    data_list.append(sys_shuff)
    idx_list.append(torch.argsort(shuff_idx).numpy())


# Instantiate aligner class
test = models.nsys_Aligner(data_list, idx_list, map_type, latent=True)

# Train
test.train(restarts, max_epoch, n_hidden, 
                gmm_scale, loss_set_scale=1, loss_cycle_scale=cycle_scale, 
                verbose=1, n_batches=n_batch, optimizer_type=optimizer, 
                learning_rate = learning_rate, 
                learn_cross_restart=restart_schedule)
