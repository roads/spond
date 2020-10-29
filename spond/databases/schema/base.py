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
"""Schema base class.

classes:
    Schema: A custom dictionary that creates a base class for working
        with TinyDB.

"""


class Schema(dict):
    """Custom dictionary object.

    Object establishes an explicit contract and has convenience methods
    for working with TinyDB.

    Subclasses must implement the attribute `schema_name`.

    Attributes:
        schema_name: A string indicating the schema name.

    """

    def __init__(self):
        """Initialize."""
        self.schema_name = 'schema'

    @property
    def w(self):
        """Return a dictionary wrapped by schema name."""
        w = {
            self.schema_name: self
        }
        return w

    def __pos__(self):
        """Return a dictionary wrapped by schema name."""
        w = {
            self.schema_name: self
        }
        return w
