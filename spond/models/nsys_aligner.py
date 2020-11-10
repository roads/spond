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
"""Models for N-system Alignment."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spond.utils as utils
import spond.datasets as datasets
import spond.losses as losses
from spond.models import MLP, Lin_map
import spond.metrics as metrics


# Alignment class
class nsys_Aligner:
    """Alignment class for mapping between N-systems."""

    def __init__(self, data_list, idx_list, map_type, latent=False,
                 sup_idx_list=None):
        """
        Initialise.

        Arguments:
            -data_list: list of point shuffled systems (tensors)
            -idx_list is a list of indices to re-map systems to
            contiguity
            -map_type is a string indicating which mapping function to use
        """
        self.data_list = [x.float() for x in data_list]
        self.idx_list = idx_list
        self.map_type = map_type
        self._n_sys = len(data_list)
        self._n_concept = data_list[0].shape[0]
        self._n_dim = data_list[0].shape[1]
        self.latent = latent

        if sup_idx_list is None:
            self.sup_idx_list = [None] * self._n_sys
        else:
            self.sup_idx_list = sup_idx_list

        if latent is False:
            self._n_mappings = (self._n_sys) * (self._n_sys - 1)
        else:
            self._n_mappings = self._n_sys

        # Check all input systems contain same number of points
        if not all(x.shape[0] == self._n_concept for x in data_list):
            raise ValueError(
                "Systems do not contain the same number of points"
                )

        # Check all input systems contain same number of dimensions
        self._n_dim = data_list[0].shape[1]
        if not all(x.shape[1] == self._n_dim for x in data_list):
            raise ValueError(
                "Systems do not have same dimensionality"
                )

    def _get_mapping_functions(self, hidden_size=100):
        """
        Create list of mapping functions for each required path.

        Create a mapping function for each path in the dense mapping
        between systems
        hidden_size: size of hidden layers in MLP
        """
        if self.map_type == "MLP":
            return (
                [MLP(self._n_dim, hidden_size).float()
                 for x in range(self._n_mappings)
                 ])

        if self.map_type == "Linear":
            return (
                [Lin_map(self._n_dim, self._n_dim).float()
                 for x in range(self._n_mappings)
                 ])

        else:
            raise ValueError(self.map_type)

    def _sort_data(self):
        """Organise each system to ensure contiguity."""
        data = [x.detach().numpy() for x in self.data_list]
        data_sort = [torch.FloatTensor(
                        [d[x] for x in self.idx_list[i]]
                    ) for i, d in enumerate(data)]
        return data_sort

    def _get_mapping_links(self, connection="dense"):
        """
        Get index list to map from a system to each other system j.

        Create a corresponding list which maps back from systems j into
        system i
        For a list of n(n-1) mappings from system i to system j,
        mappings are sorted first by system i then by system j. E.g
        in a 3 system setup, list[0] maps from system 1 to system 2,
        list[1] maps from system 1 to system 3 and list[2] maps from
        system 2 to system 1)
        These indices identify mapping functions in map_func_list, as
        functions in the list are equivalent prior to training.

        Arguments:
            connection: if "dense", return indices such that every
            mapping is performed in each direction
        """
        out_idx_list = []
        back_idx_list = []

        if connection == "dense":
            for i in range(self._n_sys):

                # Index models which map from space i to other spaces
                out_idx = list(
                    range(i*(self._n_sys-1), (i+1)*(self._n_sys-1))
                    )
                out_idx_list.append(out_idx)

                # Index models which map back from these spaces to i
                back_idx = [
                        x for x in range(self._n_mappings) if (
                            x < (self._n_sys-1) * i
                            and x % (self._n_sys-1) == (i-1)
                            )] + [
                        x for x in range(self._n_mappings) if (
                            x >= (self._n_sys-1) * (i+1)
                            and x % (self._n_sys-1) == i
                            )]
                back_idx_list.append(back_idx)

        else:
            ValueError(connection)

        return out_idx_list, back_idx_list

    def _restart_learning(self, learn_cross_restart, best_mapping_list,
                          i_restart, n_restart,
                          hidden_size=100):
        """
        Return initial list of models for restart number i_restart.

        Probabilistically initialise each model with best current
        model according to probabilities specified in schedule
        learn_cross_restart.
        If learn_cross_restart is None, no best model initialised.
        """
        new_func_list = self._get_mapping_functions(
                hidden_size=hidden_size
                )

        if learn_cross_restart is None:
            map_func_list = new_func_list

        else:
            bound = [int(t[1]*n_restart) for t in learn_cross_restart]
            current = next(x for x, v in enumerate(bound) if v > i_restart)
            thresh = [t[0] for t in learn_cross_restart][current]

            # Sample from random uniform to see where to take best
            rand = np.random.uniform(size=self._n_mappings)

            # Where rand < thresh, replace new mapping function with best
            map_func_list = ([
                best_mapping_list[i] if rand[i] < thresh
                else new_func_list[i]
                for i in range(self._n_mappings)]
            )

        return map_func_list

    def train(self, n_restart=100, max_epoch=30, hidden_size=100,
              gmm_scale=0.1, loss_set_scale=1, loss_cycle_scale=10, verbose=1,
              n_batches=1, optimizer_type="Adam", learning_rate=0.1,
              learn_cross_restart=[(0.2, 0.2), (0.7, 0.9), (1, 1)],
              loss_sup_scale=0):
        """Train aligner."""
        if self.latent is True:
            self._train_latent(n_restart, max_epoch, hidden_size,
                               gmm_scale, loss_set_scale, loss_cycle_scale,
                               verbose, n_batches, optimizer_type,
                               learning_rate, learn_cross_restart,
                               loss_sup_scale)

        if self.latent is False:
            self._train_ind(n_restart, max_epoch, hidden_size,
                            gmm_scale, loss_set_scale, loss_cycle_scale,
                            verbose, n_batches, optimizer_type, learning_rate,
                            learn_cross_restart, loss_sup_scale)

    def _train_ind(self, n_restart, max_epoch, hidden_size,
                   gmm_scale, loss_set_scale, loss_cycle_scale, verbose,
                   n_batches, optimizer_type, learning_rate,
                   learn_cross_restart, loss_sup_scale):
        """
        Training method for individual mappings between n systems.

        Training optimises cycle loss and gmm distribution loss.

        Arguments:
         - learn_cross_restart: a list of tuples specifying the restart
         - learning schedule, where tup[0] is probability of best model
           being used prior to restart tup[1]*n_restarts
        """
        # Throw error if this method is used for latent initialisation
        if self.latent is True:
            raise ValueError(self.latent)

        # Initialise list of best mapping functions
        best_mapping_list = self._get_mapping_functions(
            hidden_size=hidden_size
            )

        # Generate corresponding loss tracker
        best_loss = np.full(len(best_mapping_list), np.inf)

        # Get all sorted data for accuracy measures
        if self.idx_list is not None:
            data_sort = self._sort_data()

        # Create lists of mapping functions in/out each system
        out_idx_list, back_idx_list = self._get_mapping_links("dense")

        # Create list of system gmms
        gmm_list = []
        min_loss_list = []

        for i in range(self._n_sys):
            # Generate gmm for space i
            gmm = losses.create_gmm(self.data_list[i], gmm_scale=gmm_scale)
            gmm_list.append(gmm)

            # Save min dist loss - data as sample from own gmm
            min_loss = losses.negloglik(
                            gmm, self.data_list[i], loss_set_scale)
            min_loss_list.append(min_loss)

        for i_restart in range(n_restart):

            # Probabilistic initialisation of best models
            map_func_list = self._restart_learning(
                learn_cross_restart, best_mapping_list, i_restart, n_restart,
                hidden_size
                )

            # Create optimizer for all mapping functions
            params = [list(func.parameters()) for func in map_func_list]
            params = [y for x in params for y in x]
            optimizer = utils._get_optimizer(
                params, optimizer_type, learning_rate
                )

            # Generate index of all concepts for batching
            batching_idx = torch.randperm(self._n_concept)
            batch_size = int(np.ceil(self._n_concept/n_batches))

            for i_epoch in range(max_epoch):

                for batch in range(n_batches):
                    # batch level optimisation for pairwise cycle loss

                    if batch < n_batches-1:
                        # Select concepts for each batch
                        batch_idx = batching_idx[
                                    batch*batch_size:(batch+1)*batch_size-1
                                    ]
                    elif batch == n_batches-1:
                        batch_idx = batching_idx[
                                    batch*batch_size:self._n_concept-1
                                    ]

                    tot_cycle_loss = 0
                    for i, data in enumerate(self.data_list):
                        dat_batch = data[batch_idx]
                        dat_batch.requires_grad_()

                        # Get models which map out of space i
                        map_out = [map_func_list[x] for x in out_idx_list[i]]

                        # And corresponding models which map back to space i
                        map_back = [map_func_list[x] for x in back_idx_list[i]]

                        # Calculate cycle loss of mapping there and back
                        for j, mod in enumerate(map_out):
                            optimizer.zero_grad()
                            cycle_loss = losses.cycle_loss_flex(
                                dat_batch,
                                map_back[j].forward(
                                    mod.forward(dat_batch.float())),
                                loss_cycle_scale=loss_cycle_scale
                                )
                            cycle_loss.backward(retain_graph=True)
                            optimizer.step()
                            tot_cycle_loss += (cycle_loss.detach().numpy()
                                               / self._n_mappings)

                # Epoch level optimisation
                tot_dist_loss = 0

                # Generate accuracy tracker
                map_curr_accuracy = np.full((self._n_sys, 3), 0, dtype=float)

                # Loop through input spaces
                for i, data in enumerate(self.data_list):

                    map_out = [map_func_list[x] for x in out_idx_list[i]]
                    data.requires_grad_()

                    # And get dist loss for mapping to each output space
                    for j, gmm in enumerate(
                        [gmm for j, gmm in enumerate(gmm_list) if j != i]
                                            ):
                        optimizer.zero_grad()
                        dist_loss = torch.max(losses.negloglik(
                            gmm, map_out[j].forward(data), loss_set_scale),
                            min_loss_list[j])
                        dist_loss.backward(retain_graph=True)
                        optimizer.step()
                        tot_dist_loss += (
                            dist_loss.detach().numpy()/self._n_mappings
                                         )

                        o_dat_list = [
                            dat for j, dat in enumerate(self.data_list) if (
                                j != i
                                )
                            ]
                        # Semi-supervision
                        if (self.sup_idx_list[i] is not None) and (
                                self.sup_idx_list[j] is not None
                                ):
                            # Zero gradients for optimizer
                            optimizer.zero_grad()

                            datj = o_dat_list[j]
                            datj.requires_grad_()
                            itoj_sup = map_out[j].forward(
                                data[self.sup_idx_list[i]]
                                )
                            sup_i = losses.sup_loss_func(
                                itoj_sup, datj[self.sup_idx_list[j]],
                                loss_sup_scale
                                )

                            sup_i.backward()
                            optimizer.step()

                        # Update best_loss and best_model
                        flat_idx = (self._n_sys - 1) * i + j

                        if dist_loss.detach().numpy() < best_loss[flat_idx]:
                            best_loss[flat_idx] = dist_loss.detach().numpy()
                            best_mapping_list[flat_idx] = map_func_list[
                                flat_idx
                                ]

                        # Get mapping accuracy for this path
                        acc_f1, acc_f5, acc_f10 = (
                            metrics.mapping_accuracy(
                                map_out[j].forward(
                                    data_sort[i].detach()
                                    ).detach().numpy(),
                                data_sort[j].detach(),
                                [1, 5, 10]
                                ))
                        map_curr_accuracy[i] += np.array(
                            [acc_f1, acc_f5, acc_f10], dtype=float
                            )

                # Evaluate performance
                mean_acc = np.mean(
                    map_curr_accuracy, axis=0
                    ) / (self._n_sys - 1)

                outputs = utils.get_output(
                    i_restart, i_epoch, cycle_loss, tot_dist_loss,
                    mean_acc, verbose
                )

                if outputs is not None:
                    print(outputs[0], outputs[1])

    def _train_latent(self, n_restart, max_epoch, hidden_size,
                      gmm_scale, loss_set_scale, loss_cycle_scale, verbose,
                      n_batches, optimizer_type, learning_rate,
                      learn_cross_restart, loss_sup_scale):

        # Error if this method is used for non-latent initialisation
        if self.latent is not True:
            raise ValueError(self.latent)

        # Initialise list of best mapping functions in/out of LS
        best_mappings_out = self._get_mapping_functions(
            hidden_size=hidden_size
            )
        best_mappings_back = self._get_mapping_functions(
            hidden_size=hidden_size
            )

        # Generate corresponding loss tracker
        best_loss = np.full(self._n_mappings, np.inf)

        # Get all sorted data for accuracy measures
        if self.idx_list is not None:
            data_sort = self._sort_data()

        # Training loop
        for i_restart in range(n_restart):

            # Probabilistic initialisation of best models
            map_func_list_out = self._restart_learning(
                learn_cross_restart, best_mappings_out,
                i_restart, n_restart,
                hidden_size
                )
            map_func_list_back = self._restart_learning(
                learn_cross_restart, best_mappings_back,
                i_restart, n_restart,
                hidden_size
                )

            # Create optimizer for all mapping functions
            params_o = [list(func.parameters()) for func in map_func_list_out]
            params_o = [y for x in params_o for y in x]
            params_b = [list(func.parameters()) for func in map_func_list_back]
            params_b = [y for x in params_b for y in x]
            params = params_o + params_b

            optimizer = utils._get_optimizer(
                params, optimizer_type, learning_rate
                )

            # Generate index of all concepts for batching
            batching_idx = torch.randperm(self._n_concept)
            batch_size = int(np.ceil(self._n_concept/n_batches))

            for i_epoch in range(max_epoch):

                for batch in range(n_batches):
                    # batch level optimisation for pairwise cycle loss

                    if batch < n_batches-1:
                        # Select concepts for each batch
                        batch_idx = batching_idx[
                                    batch*batch_size:(batch+1)*batch_size-1
                                    ]
                    elif batch == n_batches-1:
                        batch_idx = batching_idx[
                                    batch*batch_size:self._n_concept-1
                                    ]

                    tot_cycle_loss = 0

                    for i, data in enumerate(self.data_list):
                        dat_batch = data[batch_idx]
                        dat_batch.requires_grad_()

                        # Calculate cycle loss of mapping in/out of LS
                        optimizer.zero_grad()
                        cycle_loss = losses.cycle_loss_flex(
                            dat_batch,
                            map_func_list_back[i].forward(
                                map_func_list_out[i](dat_batch)),
                            loss_cycle_scale=loss_cycle_scale)
                        cycle_loss.backward(retain_graph=True)
                        optimizer.step()
                        tot_cycle_loss += (cycle_loss.detach().numpy()
                                           / self._n_mappings)

                # Epoch level optimisation
                tot_dist_loss = 0

                # Generate accuracy tracker
                map_curr_accuracy = np.full((self._n_sys, 3), 0, dtype=float)

                # Loop through pairs of input spaces
                for i, data_i in enumerate(self.data_list):

                    for j, data_j in enumerate(self.data_list):

                        if j != i:

                            data_i.requires_grad_()

                            # Create gmm for latent space of i
                            gmm_i = losses.create_gmm(
                                        map_func_list_out[i].forward(data_i),
                                        gmm_scale=gmm_scale
                                        )

                            # Get minimum dist loss
                            min_loss = losses.negloglik(
                                gmm_i,
                                map_func_list_out[j].forward(data_i.detach()),
                                loss_set_scale
                                )

                            data_j.requires_grad_()

                            # Descend on loss for data_j as sample from gmm_i
                            optimizer.zero_grad()
                            dist_loss = torch.max(losses.negloglik(
                                gmm_i,
                                map_func_list_out[j].forward(data_j),
                                loss_set_scale
                                ), min_loss)
                            dist_loss.backward(retain_graph=True)
                            optimizer.step()
                            tot_dist_loss += (
                                dist_loss.detach().numpy()/self._n_mappings
                                )

                            # Get mapping accuracy for this path
                            f_x = map_func_list_back[j].forward(
                                        map_func_list_out[i].forward(
                                            data_sort[i].detach()
                                            )).detach().numpy()

                            acc_f1, acc_f5, acc_f10 = (
                                metrics.mapping_accuracy(
                                    f_x,
                                    data_sort[j].detach().numpy(),
                                    [1, 5, 10]
                                    ))
                            map_curr_accuracy[i] += np.array(
                                        [acc_f1, acc_f5, acc_f10],
                                        dtype=float
                                    )/(self._n_sys - 1)

                    # Update best_loss and best_model w full cycle loss
                    full_cycle = losses.cycle_loss_flex(
                                        data_i.detach(),
                                        map_func_list_back[i].forward(
                                            map_func_list_out[i](
                                                data_i.detach()
                                                )
                                                )
                                        )
                    if full_cycle.detach().numpy() < best_loss[i]:
                        best_loss[i] = full_cycle.detach().numpy()
                        best_mappings_out[i] = map_func_list_out[i]
                        best_mappings_back[i] = map_func_list_back[i]

                # Evaluate performance
                mean_acc = np.mean(map_curr_accuracy, axis=0)

                outputs = utils.get_output(
                    i_restart, i_epoch, cycle_loss, tot_dist_loss,
                    mean_acc, verbose
                )

                if outputs is not None:
                    print(outputs[0], outputs[1])
