import numpy as np
import importlib.util
from pathlib import Path

from serve import retrieve_b, standardize_b, unit_rows


def test_memory_statistics_and_cosine_choose_expected_row():
    memory = np.array([[0., 2.], [2., 0.], [1., 1.]], np.float32)
    ids = np.array([12, 14, 19])
    query = memory[1]
    row, score, local = retrieve_b(query, memory, ids)
    assert row == 14 and local == 1
    assert score > .99


def test_constant_dimensions_are_safe():
    memory = np.array([[0., 3.], [1., 3.], [2., 3.]], np.float32)
    mean = memory.mean(0)
    std = np.where(memory.std(0) < 1e-5, 1., memory.std(0))
    z = standardize_b(memory, mean, std)
    assert np.isfinite(unit_rows(z)).all()
    assert np.allclose(z[:, 1], 0.)


def test_unit_normalization_keeps_zero_query_finite():
    result = unit_rows(np.zeros((1, 4), np.float32))
    assert np.isfinite(result).all()
    assert np.allclose(result, 0.)


def test_paired_counts_name_candidate_rescue_in_correct_direction():
    path = Path(__file__).with_name('evaluate.py')
    spec = importlib.util.spec_from_file_location('bprior_eval_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paired = module.exact_mcnemar([1, 0, 1], [0, 1, 1])
    assert paired['candidate_only_success'] == 1
    assert paired['baseline_only_success'] == 1
    assert paired['both_success'] == 1
