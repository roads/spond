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
"""Test databases module."""

import os
import pytest
from tinydb import TinyDB, Query

import spond


def test_mid_asdict():
    """Test MethodIdentifier `asdict`."""
    method_id = 0
    hypers = {'alpha': .001, 'n_restart': 100}
    seed = 252
    mid = spond.databases.MethodIdentifier(
        method_id, hypers=hypers, seed=seed
    )

    d0 = {
        'method_id': 0,
        'hypers': {
            'alpha': .001,
            'n_restart': 100, 
        },
        'seed': 252
    }
    d1 = mid.asdict()
    assert d0 == d1


def test_record_creation(tmpdir):
    """Test benchmark record management."""
    # db = TinyDB('/Users/brett/Desktop/test.json')  # TODO
    db = TinyDB(tmpdir.join('db.json'))

    # Create method identifier.
    method_id = 0
    hypers = {'alpha': .001, 'n_restart': 100}
    seed = 252
    mid = spond.databases.MethodIdentifier(
        method_id, hypers=hypers, seed=seed
    )

    # Create benchmark result.
    benchmark_id = 0
    run = 0
    results = {
        'alignment_score': .1,
        'mapping_accuracy': .5
    }
    benchmark_result = spond.databases.BenchmarkResult(
        benchmark_id, mid, results=results
    )

    # Insert in database.
    db.insert(benchmark_result.asdict())

    # Test insert.
    s = db.search(
        (Query().method == mid.asdict()) & (Query().task == benchmark_id)
    )
    assert len(s) == 1
    assert s[0] == benchmark_result.asdict()

    # Update result of record.
    results = {
        'alignment_score': .8,
        'mapping_accuracy': .9
    }
    db.upsert(
        {'results' : results},
        (Query().method == mid.asdict()) &
        (Query().task == benchmark_id) &
        (Query().run == run)
    )

    # Test update.
    s = db.search(
        (Query().method == mid.asdict()) & (Query().task == benchmark_id)
    )
    assert len(s) == 1
    assert s[0]['results'] == results


def test_update_record():
    """Test update of existing record."""
    x = 0  # TODO
