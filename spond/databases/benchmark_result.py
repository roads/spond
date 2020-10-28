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
"""Benchmark result.

classes:
  BenchmarkResult: An object representing a benchmark result.

"""


class BenchmarkResult(object):
    """Object representing a benchmark result.
    
    Attributes:
        task_id: Integer that identifies the task.
        method_id: A MethodIdentifier object.
        results: A dictionary of results.
        run: An integer indicating the run number.

    Methods:
        asdict: Return dictionary representation of object.
        fromdict: Instaniate an object from dictionary.

    """

    def __init__(self, task_id, method_id, results={}, run=0):
        """Initialize.
        
        Arguments:
            task_id: Integer that identifies the benchmark.
            method_id: Integer that identifies the alignment method.
            results (optional): Dictionary of results.
            run (optional): An integer indicating the run. This value
                can be used to test the stochasticity of a method on a
                benchmark.

        """
        self.task_id = int(task_id)
        self.run = int(run)
        self.method_id = method_id
        self.results = results
     
    def asdict(self):
        """Return object information as dictionary."""
        d = {
            'task': self.task_id,
            'method': self.method_id.asdict(),
            'results': self.results,
            'run': self.run,
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
            A BenchmarkResult object.

        """
        return cls(**d)
