"""Small unsupervised experiment for emergent motion primitives."""

from .data import PrimitiveDataset, load_recording, make_synthetic_dataset, split_by_episode
from .experiment import ExperimentConfig, run_experiment, run_grid
from .model import PrimitiveMoE

__all__ = [
    "ExperimentConfig", "PrimitiveDataset", "PrimitiveMoE", "load_recording",
    "make_synthetic_dataset", "run_experiment", "run_grid", "split_by_episode",
]
