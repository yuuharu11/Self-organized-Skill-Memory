import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from skill_moe.skillnet.examples.memory_pilot.robocasa import latent_query as latent


def test_one_latent_token_reads_tokens_and_backpropagates():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    Net = latent._network_classes()
    net = Net(query_dim=2194, candidate_dim=60, token_dim=2048, d_model=256)
    tokens = torch.randn(3, 19, 2048)
    mask = torch.ones(3, 19, dtype=torch.bool)
    mask[1, -4:] = False
    qin = torch.randn(3, 2194)
    candidates = torch.randn(17, 60)
    scores = net(tokens, mask, qin, candidates)
    assert scores.shape == (3, 17)
    assert torch.isfinite(scores).all()
    scores.square().mean().backward()
    assert net.z0.grad is not None and torch.isfinite(net.z0.grad).all()
    assert net.log_scale.grad is not None


def test_prefix_group_ids_are_ordered_three_cameras_then_text():
    groups = latent._group_ids(num_images=3, image_tokens=4, text_tokens=5)
    np.testing.assert_array_equal(groups, [0] * 4 + [1] * 4 + [2] * 4 + [3] * 5)


def test_training_corruption_preserves_exact_input_shapes_and_reverse_xyz():
    hist = np.ones((100, 10, 12), np.float32)
    mask = np.ones((100, 10), bool)
    rng = np.random.default_rng(123)
    damaged, damaged_mask = latent._corrupt_histories(hist, mask, np.arange(100), rng)
    assert damaged.shape == hist.shape and damaged_mask.shape == mask.shape
    # All channels after xyz are never sign-flipped by the reverse-translation intervention.
    reverse_rows = np.flatnonzero((damaged[:, :, :3] == -1).all(axis=(1, 2)))
    if len(reverse_rows):
        np.testing.assert_array_equal(damaged[reverse_rows, :, 3:], 1.)


def test_episode_bootstrap_is_paired_and_reports_difference():
    lhs = np.array([2., 4., 5., 7.])
    rhs = np.array([1., 1., 2., 2.])
    task = np.array([0, 0, 1, 1])
    episode = np.array([10, 10, 20, 20])
    result = latent._paired_episode_bootstrap(lhs, rhs, task, episode, seed=1, n_boot=100)
    assert result["mean_difference"] == 3.0
    assert result["episodes"] == 2
    assert 2.0 <= result["episode_bootstrap_95"][0] <= result["episode_bootstrap_95"][1] <= 4.0


def test_eval_masks_short_units_that_have_no_first_five_action_score():
    scores = np.array([[3., 2., 1.]])
    valid = np.array([[True, True, False]])
    eligible = np.array([[True, False, False]])
    masked = latent._mask_eval_candidates(scores, valid, eligible)
    assert masked[0, 0] == 3.
    assert np.isneginf(masked[0, 1:]).all()


def test_sharded_npy_matches_concatenated_indexing():
    prefix = np.arange(24, dtype=np.float16).reshape(3, 2, 4)
    suffix = (np.arange(32, dtype=np.float16) + 100).reshape(4, 2, 4)
    joined = np.concatenate([prefix, suffix])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prefix.npy"
        s = Path(d) / "suffix.npy"
        np.save(p, prefix)
        np.save(s, suffix)
        sharded = latent._ShardedNpy(np.load(p, mmap_mode="r"), np.load(s, mmap_mode="r"), len(prefix))
        assert sharded.shape == joined.shape
        np.testing.assert_array_equal(sharded[2], joined[2])
        np.testing.assert_array_equal(sharded[3:6], joined[3:6])
        np.testing.assert_array_equal(sharded[[6, 0, 3]], joined[[6, 0, 3]])
        np.testing.assert_array_equal(sharded[4, :, 1:3], joined[4, :, 1:3])

