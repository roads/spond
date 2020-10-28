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
"""Method Identifier.

classes:
    MethodIdentifier: A method identifier.

"""

import random


class MethodIdentifier(object):
    """Object to keep track of a model's identifying information.
    
    Attributes:
        method_id: Integer that identifies the alignment method.
        hypers: Dictionary of hyperparameter values.
        seed: An integer indicated a seed value for reproducability.

    Methods:
        asdict: Return dictionary representation of object.
        fromdict: Instaniate an object from dictionary.

    """

    def __init__(self, method_id, hypers={}, seed=None):
        """Initialize.
        
        Arguments:
            method_id: Integer that identifies the alignment method.
            hypers (optional): Dictionary of hyperparameter values. The
                dictionary should list all hyperparameters that
                uniquely identify a model.
            seed (optional): Integer indicating a seed value for method
                reproducability.

        """
        self.method_id = int(method_id)
        self.hypers = hypers
        if seed is None:
            seed = random.randrange(999999)
        self.seed = seed
    
    def asdict(self):
        """Return object information as a dictionary."""
        d = {
            'method_id': self.method_id,
            'seed': self.seed,
            'hypers': self.hypers
        }
        return d

    @classmethod
    def fromdict(cls, d):
        """Create object from dictionary.

        This method is the reverse of `asdict`.

        Args:
            d: A Python dictionary, typically the output of
                `asdict`.

        Returns:
            A MethodIdentifier object.

        """
        return cls(**d)
