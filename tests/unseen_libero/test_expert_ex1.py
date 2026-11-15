import sys
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'
sys.path.insert(0, str(EXAMPLES))
import expert_ex1 as ex1


def test_nearest_rank3_excludes_query_task_and_uses_ordered_bank_row_ids():
    candidate = np.zeros((5, 10, 7), np.float32)
    candidate[0, :, 0] = 0.01  # same task, closest, must be excluded
    candidate[1, :, 0] = 0.2
    candidate[2, :, 0] = 0.4
    candidate[3, :, 0] = 0.6
    candidate[4, :, 0] = 0.8
    ids, err = ex1.nearest_unit_topk_ids(np.zeros((1, 10, 7), np.float32), candidate,
                                          np.array([4, 2, 3, 5, 6]),
                                          query_task=np.array([4]), k=3)
    assert ids.tolist() == [[1, 2, 3]]
    assert np.all(np.diff(err[0]) >= 0)


def test_ex1_training_mixture_branch_semantics():
    oracle = np.zeros((4, 3, 10, 32), np.float32)
    for rank in range(3):
        oracle[:, rank] = rank + 1
    p5 = np.full((4, 10, 32), 9, np.float32)
    out, branches = ex1.sample_training_drafts(oracle, p5, np.random.default_rng(1),
                                                probabilities=(1, 0, 0))
    assert np.all(branches == 0) and np.all(out == 1)
    out, branches = ex1.sample_training_drafts(oracle, p5, np.random.default_rng(2),
                                                probabilities=(0, 1, 0))
    assert np.all(branches == 1) and np.all((out == 2) | (out == 3))
    out, branches = ex1.sample_training_drafts(oracle, p5, np.random.default_rng(3),
                                                probabilities=(0, 0, 1))
    assert np.all(branches == 2) and np.all(out == 9)


def test_unit_drafts_copy_arm_but_take_gripper_from_pi05():
    units = np.zeros((2, 10, 7), np.float32)
    units[1, :, :6] = 3
    units[1, :, 6] = -1
    normal = np.zeros((1, 10, 32), np.float32)
    normal[:, :, 6] = .25
    result = ex1.unit_drafts(units, np.array([1]), normal)
    assert np.all(result[:, :, :6] == 3)
    assert np.all(result[:, :, 6] == .25)


def test_topk_blend_uses_natural_windows_without_transform():
    scores = np.array([[3., 2., 1.]], np.float32)
    values = np.stack([np.full((10, 7), x, np.float32) for x in (1, 4, 10)])
    blend, ids = ex1.blend_unit_windows(scores, values, k=2, temperature=1.)
    w = np.exp([3., 2.]); w /= w.sum()
    assert ids.tolist() == [[0, 1]]
    assert np.allclose(blend, w[0] * values[0] + w[1] * values[1])


def test_locked_key2_checkpoints_and_dev_blends_are_fixed():
    selected = ex1.load_locked_key2_checkpoints()
    assert selected['p5']['mode'] == 'p5'
    assert selected['p5']['selected_k'] == 20
    assert selected['p5']['temperature'] == 1.
    assert selected['e_vlm_h']['selected_k'] == 20
    assert selected['e_vj_h']['selected_k'] == 20


def test_l90x_gate_allows_only_absent_task_51(tmp_path):
    cfg=ex1.load_locked_key2_checkpoints()
    manifest={'tasks':list(range(78)),'missing_tasks':[{'task_id':51}],
              'task_row_counts':{str(i):(0 if i==51 else 1) for i in range(78)}}
    source=tmp_path/'manifest.json';source.write_text(json.dumps(manifest))
    result=ex1.write_checkpoint_manifest(tmp_path/'ex1.json',l90x_manifest=source)
    assert result['l90x_tasks_with_rows']==77
    assert result['missing_l90x_task_ids']==[51]
    manifest['missing_tasks']=[{'task_id':50}]
    source.write_text(json.dumps(manifest))
    try:
        ex1.write_checkpoint_manifest(tmp_path/'bad.json',l90x_manifest=source)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected missing task must not pass the gate')


def test_key2_candidate_order_is_joined_by_provenance_and_action_digest():
    exp3 = {
        'task': np.array([2, 1], np.int32), 'episode': np.array([20, 10], np.int32),
        'start_frame': np.array([100, 50], np.int32), 'end_frame': np.array([108, 56], np.int32),
        'length': np.array([8, 6], np.int32),
        'actions': [np.full((8, 7), .25, np.float32), np.full((6, 7), -.5, np.float32)],
    }
    source, digests = ex1.unit_source_and_digests(exp3)
    order = np.array([1, 0])
    mapping = ex1.key2_to_exp3_candidate_map(source[order], digests[order], exp3)
    np.testing.assert_array_equal(mapping, order)
    wrong = digests[order].copy()
    wrong[0] = '0' * 64
    with np.testing.assert_raises_regex(ValueError, 'fingerprint mismatch'):
        ex1.key2_to_exp3_candidate_map(source[order], wrong, exp3)


def test_candidate_action_digest_tolerates_only_sub_micro_decode_roundoff():
    left = {'task': np.array([1]), 'episode': np.array([2]), 'start_frame': np.array([3]),
            'end_frame': np.array([7]), 'length': np.array([4]),
            'actions': [np.zeros((4, 7), np.float32)]}
    right = {**left, 'actions': [np.full((4, 7), 5.96e-8, np.float32)]}
    source, digest = ex1.unit_source_and_digests(left)
    assert ex1.key2_to_exp3_candidate_map(source, digest, right).tolist() == [0]
    right['actions'] = [np.concatenate([np.zeros((4, 6), np.float32),
                                        np.full((4, 1), .000399, np.float32)], axis=1)]
    assert ex1.key2_to_exp3_candidate_map(source, digest, right).tolist() == [0]
    right['actions'] = [np.full((4, 7), 2e-6, np.float32)]
    with np.testing.assert_raises_regex(ValueError, 'fingerprint mismatch'):
        ex1.key2_to_exp3_candidate_map(source, digest, right)


def test_p5_top1_excludes_same_task_before_selection_and_records_order():
    source = np.array([[1, 1, 0, 5], [2, 2, 0, 5], [3, 3, 0, 5]], np.int64)
    digests = np.array(['a' * 64, 'b' * 64, 'c' * 64])
    scores = np.array([[100., 2., 1.], [0., 100., 9.]], np.float32)
    selected = ex1.top1_indices_with_task_exclusion(scores, source[:, 0], np.array([1, 2]))
    np.testing.assert_array_equal(selected, [1, 2])
    packed = ex1.make_p5_index_cache(np.array([20, 21]), scores, source, digests,
                                     np.array([1, 2]))
    assert str(packed['exclusion_rule'].item()) == 'same_source_task_before_top1'
    assert packed['candidate_fingerprint_sha256'].item() == ex1.candidate_bank_fingerprint(source, digests)
    np.testing.assert_array_equal(packed['selected_unit'], [1, 2])


def test_actual_retrieval_cache_keeps_candidate_arm_values_and_natural_lengths():
    source = np.array([[1, 11, 0, 5], [2, 22, 0, 5], [3, 33, 0, 5]], np.int64)
    digests = np.array(['1' * 64, '2' * 64, '3' * 64])
    scores = np.array([[100., 2., 1.]], np.float32)
    values = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    lengths = np.array([4, 3, 2], np.int32)
    packed = ex1.make_p5_index_cache(np.array([77]), scores, source, digests, np.array([1]),
                                     candidate_arm_actions=values, candidate_lengths=lengths)
    assert packed['selected_unit'].tolist() == [1]
    np.testing.assert_array_equal(packed['candidate_arm_actions'], values)
    np.testing.assert_array_equal(packed['candidate_lengths'], lengths)
    assert str(packed['exclusion_rule'].item()) == 'same_source_task_before_top1'


def test_closed_loop_candidate_values_are_reordered_to_match_key2_scores():
    exp3_windows = np.stack([np.full((10, 7), x, np.float32) for x in (10., 20., 30.)])
    key2_to_exp3 = np.array([2, 0, 1], np.int64)
    cached = ex1.reorder_candidate_windows_to_key2(exp3_windows, key2_to_exp3)
    np.testing.assert_array_equal(cached[:, 0, 0], [30., 10., 20.])
    pi = np.zeros((2, 10, 32), np.float32)
    pi[:, :, 6] = .75
    ids = np.array([[0], [2]], np.int64)
    weights = np.ones((2, 1), np.float32)
    replay = ex1.drafts_from_cached_candidate_ids(cached, ids, weights, pi)
    np.testing.assert_array_equal(replay[:, 0, 0], [30., 20.])
    np.testing.assert_array_equal(replay[:, :, 6], pi[:, :, 6])


def test_expert_output_restores_pi05_gripper_for_scoring_and_history():
    expert = np.ones((2, 10, 32), np.float32)
    pi = np.zeros_like(expert)
    pi[:, :, 6] = -.375
    restored = ex1.restore_pi05_gripper(expert, pi)
    np.testing.assert_array_equal(restored[:, :, 6], pi[:, :, 6])
    np.testing.assert_array_equal(restored[:, :, :6], expert[:, :, :6])
    np.testing.assert_array_equal(restored[:, :, 7:], expert[:, :, 7:])


def test_fit_rows_excludes_all_rows_from_the_40_fixed_learned_eval_episodes(tmp_path):
    pairs = [[i, 100 + i] for i in range(40)]
    selected_ids = list(range(621))
    selection = tmp_path / 'evaluation-selection.json'
    selection.write_text(json.dumps({'seen_demo_pairs': pairs, 'seen_row_ids': selected_ids}))
    task = np.array([pairs[i % 40][0] for i in selected_ids] + [90, 91], np.int32)
    episode = np.array([pairs[i % 40][1] for i in selected_ids] + [900, 901], np.int32)
    rows = {'task': task, 'episode': episode, 'memory': np.zeros(len(task), bool)}
    fit, meta = __import__('skill_expert_refine').fit_rows_without_learned_eval(rows, selection_path=selection)
    np.testing.assert_array_equal(fit, [621, 622])
    assert meta['fixed_eval_episodes'] == 40
    assert meta['fixed_eval_rows'] == 621
    assert meta['excluded_episode_rows'] == 621
