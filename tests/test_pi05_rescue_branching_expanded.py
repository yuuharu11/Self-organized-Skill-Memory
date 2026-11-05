from pathlib import Path
import importlib.util

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/rescue_branching/key_merge.py'
SPEC = importlib.util.spec_from_file_location('rescue_branching_key_merge', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_four_key_merge_deduplicates_and_fills_from_next_rank():
    rankings = {
        'vlm_current': [{'memory_row': x, 'score': 1.-x*.01, 'rank': i+1}
                        for i,x in enumerate([10,11,12,13,14,15,16,17])],
        'wrist_only': [{'memory_row': x, 'score': 1.-i*.01, 'rank': i+1}
                       for i,x in enumerate([10,11,20,21,22,23,24,25])],
        'codec_prior_b': [{'memory_row': x, 'score': 1.-i*.01, 'rank': i+1}
                          for i,x in enumerate([10,20,30,31,32,33,34,35])],
        'pi05_pred_to_memory_action': [{'memory_row': x, 'score': -i*.01, 'rank': i+1}
                                       for i,x in enumerate([11,20,30,40,41,42,43,44])],
    }
    merged = MODULE.merge_key_rankings(rankings)
    rows = [x['memory_row'] for x in merged]
    assert len(rows) == len(set(rows)) == 8
    assert rows[:2] == [10, 11]
    # Wrist's overlapping top two are replaced by its next distinct rows.
    assert rows[2:4] == [20, 21]
    assert merged[0]['key_ranks']['wrist_only']['rank'] == 1
    assert merged[0]['key_ranks']['codec_prior_b']['rank'] == 1


def test_action_key_is_score_audit_negative_active7_mse():
    query = np.arange(70, dtype=np.float32).reshape(10,7)/10
    actions = np.zeros((3,10,32),np.float32)
    actions[1,:,:7] = query
    actions[2,:,:7] = query + 2
    scores = MODULE.action_key_scores(query, actions)
    np.testing.assert_allclose(scores, [-np.mean(query**2), 0., -4.], rtol=0, atol=1e-7)


def test_raw_draft_is_exact_memory_gap_full_32d_action_row():
    # memory_gap/run.py constructs D-raw as np.asarray(rows['actions'][cand], np.float32).
    actions = (np.arange(4*10*32, dtype=np.float32).reshape(4,10,32)/17).astype(np.float32)
    for row in range(len(actions)):
        draft = MODULE.raw_memory_draft(actions, row)
        expected = np.asarray(actions[row], np.float32)
        assert draft.shape == (10,32)
        assert np.array_equal(draft, expected)
