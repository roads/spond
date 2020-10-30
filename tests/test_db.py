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


def test_strategy_init():
    """Test Strategy."""
    strat_id = 13
    hypers = {'alpha': .001, 'n_restart': 100}
    seed = 252
    strategy = spond.databases.Strategy(
        strat_id, hypers=hypers, seed=seed
    )

    assert strategy['id'] == 13
    assert strategy['hypers'] == {
        'alpha': .001,
        'n_restart': 100,
    }
    assert strategy['seed'] == 252

    assert +strategy == {
        'strategy': {
            'id': 13,
            'hypers': {
                'alpha': .001,
                'n_restart': 100,
            },
            'seed': 252
        }
    }


def test_task_init():
    """Test Task."""
    # Test Task without optional arguments.
    task_id = 0
    task = spond.databases.Task(
        task_id
    )

    assert task['id'] == task_id
    assert task['rerun'] == 0

    # Test Task with optional arguments.
    task_id = 12
    task = spond.databases.Task(
        task_id,
        rerun=1
    )

    assert task['id'] == 12
    assert task['rerun'] == 1

    assert +task == {
        'task': {
            'id': 12,
            'rerun': 1
        }
    }


def test_result_init():
    """Test Task."""
    # Test Task without optional arguments.
    results = spond.databases.Results(
        {'alignment_score': .2}
    )

    assert results['alignment_score'] == .2

    assert +results == {
        'results': {
            'alignment_score': .2
        }
    }


def test_insert(tmpdir):
    """Test benchmark record management."""
    db = TinyDB(tmpdir.join('db.json'))

    # Create Strategy.
    strat_id = 0
    hypers = {'alpha': .001, 'n_restart': 100}
    seed = 252
    strategy = spond.databases.Strategy(
        strat_id, hypers=hypers, seed=seed
    )

    # Create Task.
    task_id = 0
    rerun = 1
    task = spond.databases.Task(
        task_id, rerun=rerun
    )

    # Create results.
    results = spond.databases.Results(
        {
            'alignment_score': .1,
            'mapping_accuracy': .5
        }
    )

    tsr = {**+task, **+strategy, **+results}

    # Insert in database.
    db.insert(tsr)

    # Test basic insert.
    s = db.search(
        (Query().task == task) & (Query().strategy == strategy)
    )
    assert len(s) == 1
    assert s[0] == tsr


def test_upsert(tmpdir):
    """Test upsert."""
    db = TinyDB(tmpdir.join('db.json'))

    # Create Strategy.
    strat_id = 0
    hypers = {'alpha': .001, 'n_restart': 100}
    seed = 252
    strategy = spond.databases.Strategy(
        strat_id, hypers=hypers, seed=seed
    )

    # Create Task.
    task_id = 0
    rerun = 0
    task = spond.databases.Task(
        task_id, rerun=rerun
    )

    # Create results.
    results = spond.databases.Results(
        {
            'alignment_score': .1,
            'mapping_accuracy': .5
        }
    )

    # Insert in database.
    tsr = {**+task, **+strategy, **+results}
    db.upsert(
        tsr,
        (Query().task == task) & (Query().strategy == strategy)
    )

    # Test insert upsert.
    s = db.search(
        (Query().task == task) & (Query().strategy == strategy)
    )
    assert len(s) == 1
    assert s[0] == tsr

    # Update result of record.
    results = spond.databases.Results(
        {
            'alignment_score': .8,
            'mapping_accuracy': .9
        }
    )
    tsr = {**+task, **+strategy, **+results}
    db.upsert(
        tsr,
        (Query().task == task) &
        (Query().strategy == strategy)
    )

    # Test update upsert.
    s = db.search(
        (Query().task == task) & (Query().strategy == strategy)
    )
    assert len(s) == 1
    assert s[0]['results'] == results
