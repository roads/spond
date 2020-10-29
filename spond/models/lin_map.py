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

"""Lin_map
Classes:
    Lin_map: linear mapping function
"""

import torch
import torch.nn as nn

class Lin_map(nn.Module):
    """
    Linear mapping from one space to another
        Parameters:
            n_dim_in: dimensionality of input
            n_dim_out: dimensionality of output
        
    """

    def __init__(self, n_dim_in, n_dim_out):
        super(Lin_map, self).__init__()
        
        self.fc1 = nn.Linear(n_dim_in, n_dim_out)

        # Initialise with Glorot uniform
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        """Feed-forward pass."""
        y = self.fc1(x)
        return y
