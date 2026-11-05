import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "skill_moe/skillnet/examples/memory_pilot/pi05_memory/key_mechanism"
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location("key_mechanism_run", CODE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def _fake_rows():
    # Deliberately choose dataset row IDs with a different order from pool positions.
    actions = np.zeros((9, 10, 7), np.float32)
    for i in range(9):
        actions[i] = i / 10
    task = np.array([0, 10, 10, 11, 11, 12, 12, 13, 13], np.int32)
    return {"actions": actions, "task": task}


def test_retrieve_scores_actions_by_candidate_row_id_and_excludes_same_task():
    rows = _fake_rows()
    qids = np.array([0, 5])
    mids = np.array([8, 2, 6, 1, 7, 3])
    qfeat = np.array([[1., 0.], [0., 1.]], np.float32)
    mfeat = np.array([[0., 1.], [1., 0.], [0., .9], [.8, .2], [0., .8], [.7, .3]], np.float32)
    cand, score, mse = run._retrieve(qfeat, mfeat, qids, mids, rows, rows["actions"], k=3)
    assert cand[0, 0] == 2  # highest score but same task? query task 0, so allowed
    assert 6 not in cand[1]  # query task 12 excluded
    expected = np.mean((rows["actions"][qids][:, None, :, :7] - rows["actions"][cand][..., :7]) ** 2, axis=(-1, -2))
    np.testing.assert_allclose(mse, expected, rtol=0, atol=1e-7)
    # Explicitly catch the former pool-position / dataset-row-ID confusion.
    wrong = np.mean((rows["actions"][qids[0]][None, :, :7] - rows["actions"][np.arange(3)][..., :7]) ** 2, axis=(-1, -2))
    assert not np.allclose(mse[0], wrong)
    assert np.all(np.diff(score[0]) <= 0)


def test_holdout_tasks_are_suite_balanced_and_long_is_rejected():
    hold = run.choose_holdout_tasks()
    assert hold.tolist() == [17, 19, 28, 29, 35, 38]
    assert sum(10 <= x < 20 for x in hold) == 2
    assert sum(20 <= x < 30 for x in hold) == 2
    assert sum(30 <= x < 40 for x in hold) == 2
    rows = {"task": np.array([9, 10, 39], np.int32)}
    with pytest.raises(AssertionError):
        run.assert_no_long(np.array([0]), rows, "bad")
    run.assert_no_long(np.array([1, 2]), rows, "good")


def test_attention_models_shapes_and_masked_attention():
    torch.set_num_threads(1)
    from models import M1AttentionKey, M2SalientTokens, M3CandidateJudge
    tokens = torch.randn(2, 12, 2048)
    context = torch.randn(2, 79)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -2:] = False
    m1 = M1AttentionKey(slots=3, dim=24, token_count=12, heads=4)
    key, _, att = m1.encode_tokens(tokens, context, mask)
    assert key.shape == (2, 24)
    assert att.shape == (2, 3, 12)
    assert torch.all(att[:, :, -2:] == 0)
    m2 = M2SalientTokens(dim=24, token_count=12)
    key2, _, weights = m2.encode_tokens(tokens, context, mask)
    assert key2.shape == (2, 24)
    assert weights.shape == (2, 12)
    assert torch.all(weights[:, -2:] == 0)
    judge = M3CandidateJudge(dim=24, heads=4)
    score = judge(torch.randn(2, 12, 24), mask, torch.randn(2, 5, 10, 7), torch.randn(2, 5, 8))
    assert score.shape == (2, 5)


def test_cpu_state_dict_snapshot_keeps_live_model_on_device():
    from models import M1AttentionKey
    model = M1AttentionKey(slots=3, dim=24, token_count=12, heads=4)
    model.to(run.DEVICE)
    before = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    snapshot = run._cpu_state_dict(model)
    # The live model must stay where it was so post-save encoding still hits DEVICE.
    assert next(model.parameters()).device.type == torch.device(run.DEVICE).type
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), value)
        assert snapshot[key].device.type == 'cpu'
        torch.testing.assert_close(snapshot[key], value.cpu())


def test_gap_threshold_reads_corrected_then_summary(tmp_path):
    (tmp_path / 'summary_corrected.json').write_text(json.dumps({'threshold': 0.02}))
    (tmp_path / 'summary.json').write_text(json.dumps({'threshold': 0.05}))
    assert run._gap_threshold(tmp_path) == 0.02
    (tmp_path / 'summary_corrected.json').unlink()
    assert run._gap_threshold(tmp_path) == 0.05
    (tmp_path / 'summary.json').unlink()
    with pytest.raises(FileNotFoundError):
        run._gap_threshold(tmp_path)
