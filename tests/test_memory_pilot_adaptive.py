import importlib.util
from pathlib import Path

import numpy as np


SERVE = Path('/work/skill_moe/skillnet/examples/memory_pilot/pi05_memory/adaptive/serve.py')
SPEC = importlib.util.spec_from_file_location('adaptive_serve_for_test', SERVE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_score_schedule_including_exact_edges():
    values = [0.929679, 0.92968, 0.942669, 0.94267, 0.950019, 0.95002]
    adaptive = [MODULE.score_mode(x, 'adaptive') for x in values]
    with_fallback = [MODULE.score_mode(x, 'adaptive_fb') for x in values]
    assert adaptive == [1, 1, 1, 2, 2, 3]
    assert with_fallback == [0, 1, 1, 2, 2, 3]


def test_score_modes_reject_unknown_condition():
    try:
        MODULE.score_mode(.94, 'unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown schedule condition was accepted')


def test_rng_seed_is_request_local_and_order_independent():
    requests = [{'s2_rng_seed': 700001}, {'s2_rng_seed': 800003}, {'s2_rng_seed': 700001}]
    outputs = {request['s2_rng_seed']: MODULE.observation_seed(request)
               for request in requests}
    assert outputs[700001] == 700001
    assert outputs[800003] == 800003
    assert MODULE.observation_seed(requests[2]) == MODULE.observation_seed(requests[0])


def test_schedules_keep_the_predeclared_edge_values():
    np.testing.assert_array_equal(MODULE.SCORE_EDGES, [0.92968, 0.94267, 0.95002])
    assert MODULE.MODES == {'adaptive': [1, 1, 2, 3], 'adaptive_fb': [0, 1, 2, 3]}


def test_blockwise_fp16_key_normalization_matches_full_bank_formula():
    import torch
    torch.manual_seed(4)
    keys = torch.randn(19, 7, 11).half()
    expected = keys.clone()
    expected /= torch.linalg.vector_norm(expected.float(), dim=-1,
                                         keepdim=True).clamp_min(1e-8).half()
    actual = MODULE.normalize_token_keys_blockwise(keys.clone(), torch, block_rows=4)
    assert torch.equal(actual, expected)
