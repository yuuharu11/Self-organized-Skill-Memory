import numpy as np

from skill_moe.skillnet.examples.emergent.light.intermediate import selection_review as sr


def test_member_lse_prefers_relevant_member_and_size_penalty():
    keys = np.asarray([[1., 0.], [.99, .1], [0., 1.], [0., .9]], np.float32)
    ids = np.asarray([0, 0, 1, 1])
    score = sr.member_lse_scores(np.asarray([1., 0.]), keys, ids, 2, r=1, gamma=0)
    assert score.argmax() == 0
    # Duplicating identical members raises uncorrected log-sum-exp, then gamma
    # removes the log cardinality contribution.
    dup = np.concatenate([keys, keys[:2]])
    dup_ids = np.r_[ids, [0, 0]]
    raw = sr.member_lse_scores([1., 0.], dup, dup_ids, 2, r=5, gamma=0)
    corrected = sr.member_lse_scores([1., 0.], dup, dup_ids, 2, r=5, gamma=1)
    assert raw[0] > corrected[0]


def test_bbse_recovers_label_shift_and_corrects_probabilities():
    confusion = np.asarray([[.9, .1], [.2, .8]])
    prior = np.asarray([.25, .75])
    predicted_marginal = prior @ confusion
    probs = np.repeat(predicted_marginal[None], 100, axis=0)
    estimated, adjusted = sr.bbse_priors(confusion, probs, np.asarray([.5, .5]))
    assert np.allclose(estimated, prior, atol=1e-3)
    assert adjusted.shape == probs.shape
    assert np.allclose(adjusted.sum(1), 1.0)
    hard = np.repeat(np.asarray([[1., 0.]]) * .375 + np.asarray([[0., 1.]]) * .625, 100, axis=0)
    estimated_hard, _ = sr.bbse_priors(confusion, probs, np.asarray([.5, .5]), marginal_source=hard)
    assert np.allclose(estimated_hard, prior, atol=1e-3)


def test_kappa_and_interval_grouping():
    assert sr.cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1], 2) == 1.
    assert sr.cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0], 2) < 0
    groups = sr.interval_groups(["a"] * 5, [0, 1, 2, 3, 4], [-1, -1, 1, 1, -1])
    assert [len(g) for g in groups] == [2, 2, 1]


def test_confusion_merge_obeys_threshold_and_functional_distance():
    confusion = np.asarray([[0., .8, 0.], [.7, 0., 0.], [0., 0., 0.]])
    distances = np.asarray([[0., .1, .8], [.1, 0., .7], [.8, .7, 0.]])
    mapping = sr._merge_confused(confusion, .5, distances, .2)
    assert mapping[0] == mapping[1]
    assert mapping[2] != mapping[0]


def test_functional_and_chance_metrics_use_interval_units():
    metrics = sr.selection_metrics([0, 0, 1, 1], [0, 1, 1, 1], [{0}, {0, 1}, {1}, set()], 2)
    assert metrics["strict_top1"] == .75
    assert metrics["functional_accuracy"] == .75
    assert metrics["functional_coverage"] == .75
    assert metrics["majority_chance"] == .5
    assert metrics["frequency_chance"] == .5
