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
"""Models for Alignment."""
import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from spond.losses import cycle_loss_func, sup_loss_func, set_loss_func
from spond.metrics import mapping_accuracy
from spond.models import MLP

# TODO: make mapping function customizable
class AlignmentDataset(Dataset):
    """PyTorch dataset module to manage batch processing for cycle loss."""

    def __init__(self, x, y):
        """Initialize."""
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        """Return length."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return items based on indices."""
        x_batch = self.x[index]
        y_batch = self.y[index]

        return x_batch, y_batch


class Aligner:
    """Experiment class for registerintg data and training configurations."""

    def __init__(self, x, y, y_idx_map, sup_idx_x=None, sup_idx_y=None):
        """Initialize."""
        self._n_concept = x.shape[0]
        self._n_dim = x.shape[1]
        self._y_idx_map = y_idx_map
        self._sup_idx_x = sup_idx_x
        self._sup_idx_y = sup_idx_y
        self.dataset = AlignmentDataset(x, y)
        

    def train(self, n_restart=100, max_epoch=30, hidden_size=100, gmm_scale=0.1, loss_set_scale=1.0, loss_cycle_scale=10, loss_sup_scale=10):
        """Main training routine."""
        # logging template
        template_loss = 'Restart {0} Loss | total: {1:.5g} | cycle: {2:.3g} | f_set: {3:.3g} | g_set: {4:.3g} | f_sup: {5:.3g} | g_sup: {6:.3g}'
        template_acc = '{0} Accuracy | 1: {1:.2f} | 5: {2:.2f} | 10: {3:.2f} | half: {4:.2f}'

        # Set up dataloader
        batch_size = 100
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # TODO: discuss the distinction fo variables
        x = self.dataset.x
        y = self.dataset.y

        gmm_x_samples = self.dataset.x
        gmm_y_samples = self.dataset.y

        best_model_f = MLP(self._n_dim, hidden_size)
        best_model_g = MLP(self._n_dim, hidden_size)

        set_loss_f_best = np.inf
        set_loss_g_best = np.inf

        for i_restart in range(n_restart):
            # Use last few restarts to fine-tune best model.
            thresh = .7  # TODO
            if i_restart < (n_restart - 10):  # TODO
                rand_val = np.random.rand(2)
                if rand_val[0] < thresh:
                    model_f = MLP(self._n_dim, hidden_size)
                    # print('\tNew model_f')
                else:
                    model_f = best_model_f

                if rand_val[1] < thresh:
                    model_g = MLP(self._n_dim, hidden_size)
                    # print('\tNew model_g')
                else:
                    model_g = best_model_g
            else:
                model_f = best_model_f
                model_g = best_model_g

            # TODO: incldue GMM params
            params = list(model_f.parameters()) + list(model_g.parameters())
            # TODO: customizable optimizer, put initial learning rate into the config file
            optimizer = optim.Adam(params)

            for i_epoch in range(max_epoch):
                for x_batch, y_batch in dataloader:
                    # batch level optimization for pairwise cycle loss
                    optimizer.zero_grad()

                    x_c = model_g(model_f(x_batch))
                    y_c = model_f(model_g(y_batch))

                    cycle_loss = cycle_loss_func(x_batch, y_batch, x_c, y_c, loss_cycle_scale)
                    cycle_loss.backward()
                    optimizer.step()

                # epoch level optimization
                optimizer.zero_grad()
                
                # set loss at the distribution level
                f_x = model_f(x)
                g_y = model_g(y)
                set_loss_f = set_loss_func(f_x, gmm_y_samples, loss_set_scale)
                set_loss_g = set_loss_func(g_y, gmm_x_samples, loss_set_scale)

                # Semi-supervision for a subset of indices
                if (self._sup_idx_x is not None) and (self._sup_idx_y is not None):
                    # between y and f_x
                    f_x_sup = model_f(x[self._sup_idx_x])
                    sup_loss_f = sup_loss_func(f_x_sup, y[self._sup_idx_y], loss_sup_scale)

                    # between x and g_y
                    g_y_sup = model_g(y[self._sup_idx_y])
                    sup_loss_g = sup_loss_func(g_y_sup, x[self._sup_idx_x], loss_sup_scale)
                else:
                    sup_loss_f, sup_loss_g = 0, 0

                total_loss = sum([set_loss_f, set_loss_g, sup_loss_f, sup_loss_g])
                total_loss.backward()
                optimizer.step()

                if self._y_idx_map is not None:
                    acc_f1, acc_f5, acc_f10, acc_fhalf = mapping_accuracy(
                        f_x.detach().cpu().numpy(),
                        y[self._y_idx_map].detach().cpu().numpy()
                    )
                    acc_g1, acc_g5, acc_g10, acc_ghalf = mapping_accuracy(
                        g_y[self._y_idx_map].detach().cpu().numpy(),
                        x.detach().cpu().numpy()
                    )

            # Compare against the current best loss
            if set_loss_f.item() < set_loss_f_best:
                set_loss_f_best = set_loss_f.item()
                best_model_f = model_f
                # print('\tBeat best model_f.')

            if set_loss_g.item() < set_loss_g_best:
                set_loss_g_best = set_loss_g.item()
                best_model_g = model_g
                # print('\tBeat best model_g.')

            # Evaluate performance. TODO: better logging
            if i_restart % 1 == 0:
                print(template_loss.format(
                        i_restart+1,
                        total_loss.item() + cycle_loss.item(),
                        cycle_loss.item(),
                        set_loss_f.item(),
                        set_loss_g.item(),
                        sup_loss_f,
                        sup_loss_g
                      ) + 
                      template_acc.format('\nf(x)', acc_f1, acc_f5, acc_f10, acc_fhalf) +
                      template_acc.format('\tg(y)', acc_g1, acc_g5, acc_g10, acc_ghalf)
                      )

                if acc_f1 > .95 and acc_g1 > .95:
                    break
