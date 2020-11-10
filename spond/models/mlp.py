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
"""Multilayer Perceptron.

Clases:
    MLP: Multiplayer Perceptron.

"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer Perceptron to map from one system to another."""

    def __init__(self, input_size, hidden_size):
        """Initialize."""
        super(MLP, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        self.fc1 = nn.Linear(self._input_size, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self.fc3 = nn.Linear(self._hidden_size, self._hidden_size)
        self.fc4 = nn.Linear(self._hidden_size, self._input_size)

        # Glorot uniform initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        """Forward."""
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        out = torch.tanh(self.fc4(x))
        return out
