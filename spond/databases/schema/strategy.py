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
"""Strategy identifier.

classes:
    Strategy: An object for uniquely identifying a method or model.

"""

import random

from spond.databases.schema.base import Schema


class Strategy(Schema):
    """Object to keep track of a strategy's identifying information.

    The resulting dictionary has the following keys and semantics:
        id: Integer that identifies the strategy.
        hypers: Dictionary of hyperparameter values.
        seed: An integer indicated a seed value for reproducability.

    """

    def __init__(self, id, hypers={}, seed=None):
        """Initialize.

        Arguments:
            id: Integer that identifies the strategy.
            hypers (optional): Dictionary of hyperparameter values. The
                dictionary should list all hyperparameters that
                uniquely identify a particular strategy.
            seed (optional): Integer indicating a seed value for
                reproducability.

        """
        self.schema_name = 'strategy'
        if seed is None:
            seed = random.randrange(999999)
        self.update(
            {
                'id': int(id),
                'hypers': hypers,
                'seed': seed
            }
        )
