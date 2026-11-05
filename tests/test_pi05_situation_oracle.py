import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/situation_oracle'))
from core import (align_subsequence, assert_nonlong_roles, choose_action_aligned_demo,
                  compact_positions_to_row_ids, topk_row_ids)


def test_compact_positions_are_not_dataset_row_ids():
    pool = np.asarray([901, 17, 7002, 44], np.int64)
    pos = np.asarray([2, 0, 3], np.int64)
    assert compact_positions_to_row_ids(pos, pool).tolist() == [7002, 901, 44]
    assert topk_row_ids(np.asarray([.1, .8, .2, .9]), pool, 3).tolist() == [44, 17, 7002]
    with pytest.raises(IndexError):
        compact_positions_to_row_ids([4], pool)


def test_action_alignment_accounts_only_for_removed_hdf5_frames():
    # source omitted frame 2 (a no-op); keep original HDF5 frame numbers.
    demo = np.asarray([[0.], [1.], [0.], [2.], [3.], [4.]], np.float32)
    source = demo[[0, 1, 3, 4, 5]]
    mapping, matched = align_subsequence(source, demo)
    assert matched == len(source)
    assert mapping.tolist() == [0, 1, 3, 4, 5]
    name, selected = choose_action_aligned_demo(source, {'demo_1': demo})
    assert name == 'demo_1'
    np.testing.assert_array_equal(selected, mapping)


def test_action_alignment_fails_closed_on_mismatch_or_ambiguity():
    source = np.asarray([[0.], [2.]], np.float32)
    demo = np.asarray([[0.], [1.], [2.]], np.float32)
    with pytest.raises(ValueError, match='no HDF5 demo'):
        choose_action_aligned_demo(source, {'bad': np.asarray([[9.], [8.]], np.float32)})
    with pytest.raises(ValueError, match='ambiguous'):
        choose_action_aligned_demo(source, {'a': demo, 'b': demo.copy()})


def test_fit_and_selection_ids_reject_long_dataset_rows():
    tasks = np.asarray([0, 9, 10, 11, 39], np.int32)
    assert_nonlong_roles(tasks, {'pool': [2, 3, 4], 'dev': [2, 4]})
    with pytest.raises(AssertionError, match='Long rows'):
        assert_nonlong_roles(tasks, {'selection': [0, 2]})

