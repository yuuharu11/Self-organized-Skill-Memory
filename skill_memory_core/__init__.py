"""Portable action-memory retrieval and bridge primitives.

Importing this package does not import a robotics stack or an accelerator
framework. The OpenPI adapter is imported only when explicitly requested.
"""

from .bridge import bridge_batch, integrate_euler
from .drafts import sample_rank_mixture, shuffle_cross_task
from .key import Retriever, SalientTokenKey, build_context
from .memory import MotionMemory, cut_fixed_windows
from .metrics import active_mse, episode_bootstrap, wilson_interval
from .units import RawFunctional, m6_fit, medoid

__all__ = [
    "MotionMemory", "cut_fixed_windows", "SalientTokenKey", "build_context",
    "Retriever", "sample_rank_mixture", "shuffle_cross_task", "bridge_batch",
    "integrate_euler", "RawFunctional", "m6_fit", "medoid", "active_mse",
    "episode_bootstrap", "wilson_interval",
]
