"""CPU-sized intervention on the actual skill cross-attention module.

This establishes an architectural property, not a released-policy success rate.
Run with JAX_PLATFORMS=cpu to avoid competing with active evaluations.
"""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill_moe/skillnet/src"))
from openpi.models.gemma_moe_skill import MoEFeedForward


def test_skill_order_without_positions_is_invariant_but_content_can_matter():
    rng = np.random.default_rng(20261102)
    x = jnp.asarray(rng.normal(size=(1, 7, 8)), jnp.float32)
    skills = jnp.asarray(rng.normal(size=(1, 5, 8)), jnp.float32)
    mask = jnp.asarray([[True, True, True, False, False]])
    module = MoEFeedForward(num_experts=4, hidden_dim=16, output_dim=8)
    weights = module.init(jax.random.key(7), x, skills, mask)
    original = module.apply(weights, x, skills, mask)
    permutation = jnp.asarray([2, 0, 4, 1, 3])
    reordered = module.apply(weights, x, skills[:, permutation], mask[:, permutation])
    changed = module.apply(weights, x, skills.at[:, 0].set(-3 * skills[:, 0]), mask)
    np.testing.assert_allclose(original[0], reordered[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(original[2], reordered[2])
    assert float(jnp.max(jnp.abs(original[0] - changed[0]))) > 1e-5
