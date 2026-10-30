import numpy as np
import torch

from skill_moe.skillnet.examples.emergent.light.intermediate.skill_v3 import (
    LatentSkillBank,
    SceneConditionedVAE,
    evaluate_assignments,
)


def test_scene_conditioned_decoder_uses_only_latent_plus_context_on_decode():
    model = SceneConditionedVAE(4, 6, 3, 25 * 7)
    curve = torch.randn(2, 32, 7)
    mu, _ = model.encode(curve)
    key = torch.randn(2, 6)
    state = torch.randn(2, 3)
    phase = torch.tensor([0.2, 0.7])
    out = model.decode(mu, key, state, phase)
    assert out.shape == (2, 25 * 7)
    changed = model.decode(mu, key + 1, state, phase)
    assert not torch.allclose(out, changed)


def test_open_ended_latent_bank_finds_supported_motion_groups_and_filters_outlier():
    rng = np.random.default_rng(4)
    z, segments = [], []
    for group, center in enumerate((0.0, 2.0)):
        for i in range(15):
            z.append(rng.normal(center, 0.01, size=4))
            segments.append({"episode_id": f"ep{group}-{i % 3}"})
    z.append(np.asarray([12., 12., 12., 12.]))
    segments.append({"episode_id": "one-off"})
    bank = LatentSkillBank(kappa=0.1, min_support=10, min_episodes=3).fit(
        np.asarray(z, np.float32), segments, seed=7)
    assert bank.centers.shape == (2, 4)
    assert np.all(np.bincount(bank.assignments[:30], minlength=2) >= 10)
    assert bank.epsilon > 0


def test_external_assignment_evaluator_accepts_replaceable_skill_anchors():
    query = np.asarray([[1., 0.], [0., 1.]], np.float32)
    prototypes = {"keys": np.eye(2, dtype=np.float32)}
    built = []

    def anchor_builder(unit_ids, row_indices):
        built.append((unit_ids.copy(), row_indices.copy()))
        return np.stack([np.full((2, 7), float(uid), np.float32) for uid in unit_ids])

    result = evaluate_assignments(
        rows={}, query_keys=query, assignments=np.asarray([0, 1]),
        prototypes=prototypes, anchor_builder=anchor_builder,
        truth_assignments=np.asarray([0, 1]))
    assert result["selection"]["top1_accuracy"] == 1.0
    assert result["selection"]["top5_accuracy"] == 1.0
    assert result["anchors"].shape == (2, 2, 7)
    assert np.array_equal(built[0][0], [0, 1])
