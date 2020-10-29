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
"""Task Identifier.

classes:
    Task: A task identifier.

"""

from spond.databases.schema.base import Schema


class Task(Schema):
    """Object to keep track of a task's identifying information.

    The resulting dictionary has the following keys and semantics:
        id: Integer that identifies the task.
        rerun: Integer that identifiers the rerun of the task.

    """

    def __init__(self, id, rerun=0):
        """Initialize.

        Arguments:
            id: Integer that identifies the task.
            rerun (optional): Integer that identifies the rerun of the
                task. This value can be used to test the stochasticity
                of a particular strategy.

        """
        self.schema_name = 'task'
        self.update(
            {
                'id': int(id),
                'rerun': int(rerun)
            }
        )
