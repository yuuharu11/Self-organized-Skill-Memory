"""Shared held-out-suite definitions and row selection for action experiments."""
from __future__ import annotations


SPLITS = {
    "A": (tuple(range(20, 40)), tuple(range(10, 20)), "object+spatial → goal"),
    "B": (tuple(range(10, 30)), tuple(range(30, 40)), "object+goal → spatial"),
}


def subset_rows(rows, mask):
    """Select the same rows from every array, preserving their order and dtype."""
    return {name: value[mask] for name, value in rows.items()}
