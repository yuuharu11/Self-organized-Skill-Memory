#!/usr/bin/env python3
"""From demos to the draft handed to denoising, using variable-length DCT motion units.

    SkillMemory.build   cut demos, remove direction and size, merge under an error tolerance
    DraftPlanner        hold the current unit and its phase; each query returns one draft

The planner takes two parts it does not decide itself. `select(frame)` names the next
unit, and `reference` supplies direction and size for the frames of the draft. Offline
evaluation passes the correct actions; a running policy passes retrieval and a policy
output. The returned draft is normalized [horizon, 7], before noise is mixed in.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import idct

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dct_tolerance_merge as merge  # noqa: E402
import variable_shape_memory as vs  # noqa: E402

FORM = 'rms'
HORIZON = 10


@dataclass
class SkillMemory:
    values: np.ndarray  # [units, K*6 + grip_k], independent of length
    lengths: np.ndarray  # [units], the length of the real segment behind each unit
    grip_k: int

    @classmethod
    def build(cls, traces, *, segment, tau):
        """`traces` are normalized [T,7] demos; `segment` is 'fixed10' or 'e<epsilon>'."""
        grip_k = vs.grip_values(segment)
        values, matrices, lengths = [], [], []
        for trace in traces:
            for start, end in vs.spans_of(trace, segment):
                values.append(vs.encode(trace, start, end, FORM, grip_k))
                matrices.append(merge.pose_matrix(trace[start:end, :6], FORM))
                lengths.append(end - start)
        values, matrices, lengths = np.stack(values), np.stack(matrices), np.asarray(lengths)
        units, _ = merge.greedy_cover(merge.rebuild_errors(values, matrices, grip_k) <= tau)
        return cls(values[units], lengths[units], grip_k)

    def save(self, path):
        np.savez_compressed(path, values=self.values, lengths=self.lengths, grip_k=self.grip_k)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z['values'], z['lengths'], int(z['grip_k']))

    def shape(self, unit):
        """The unit at its own length: [L,6] shape and [L] normalized gripper."""
        length = int(self.lengths[unit])
        value = self.values[unit]
        keep = min(vs.K, length)
        c = np.zeros((length, 6))
        c[:keep] = value[:vs.K * 6].reshape(vs.K, 6)[:keep] * np.sqrt(length)
        g = np.zeros(length)
        keep = min(self.grip_k, length)
        g[:keep] = value[vs.K * 6:vs.K * 6 + keep] * np.sqrt(length)
        return idct(c, type=2, norm='ortho', axis=0), idct(g, type=2, norm='ortho')


def align(shape, reference):
    """Give a [n,6] shape the net direction and RMS of a normalized [n,6] reference."""
    target = vs.to_physical(reference)
    source_rotation = np.asarray(vs.rotation_to_x(shape[:, :3].sum(0)), np.float64)
    target_rotation = np.asarray(vs.rotation_to_x(target[:, :3].sum(0)), np.float64)
    turn = source_rotation.T @ target_rotation
    source_rms = float(np.sqrt(np.mean(shape ** 2)))
    size = float(np.sqrt(np.mean(target ** 2))) / max(source_rms, 1e-8)
    moved = np.column_stack([shape[:, :3] @ turn, shape[:, 3:6] @ turn]) * size
    return vs.to_normalized(moved)


class DraftPlanner:
    """Follows one unit to its end before asking for the next."""

    def __init__(self, memory, select, *, horizon=HORIZON, unit_reference=None):
        """With `unit_reference(first, length)`, direction and size are set once per unit."""
        self.memory, self.select, self.horizon = memory, select, horizon
        self.unit_reference = unit_reference
        self.frame = 0
        self.plan = []  # (unit, first frame of that unit), oldest first
        self.selections = 0

    def _ensure(self, last_frame):
        while not self.plan or self._end(self.plan[-1]) <= last_frame:
            start = self._end(self.plan[-1]) if self.plan else self.frame
            self.plan.append((int(self.select(start)), start))
            self.selections += 1

    def _end(self, item):
        return item[1] + int(self.memory.lengths[item[0]])

    def draft(self, reference):
        """Return the next [horizon,7] draft; `reference` is normalized [horizon,>=6]."""
        self._ensure(self.frame + self.horizon - 1)
        out = np.empty((self.horizon, 7), np.float64)
        stop = self.frame + self.horizon
        for unit, first in self.plan:
            lo, hi = max(first, self.frame), min(self._end((unit, first)), stop)
            if lo >= hi:
                continue
            shape, grip = self.memory.shape(unit)
            a, b = lo - self.frame, hi - self.frame
            if self.unit_reference is None:
                out[a:b, :6] = align(shape[lo - first:hi - first], reference[a:b, :6])
            else:
                whole = self.unit_reference(first, len(shape))
                out[a:b, :6] = align(shape[:len(whole)], whole)[lo - first:hi - first]
            out[a:b, 6] = grip[lo - first:hi - first]
        return out

    def advance(self, steps):
        self.frame += steps
        self.plan = [p for p in self.plan if self._end(p) > self.frame]


def oracle_selector(memory, trace, tau, *, match_gripper=False):
    """Pick the longest unit that rebuilds the next frames within tau, or else the closest.

    With `match_gripper`, a unit whose gripper sign differs in any covered frame cannot fit.
    """
    by_length = {int(n): np.flatnonzero(memory.lengths == n) for n in np.unique(memory.lengths)}

    def select(frame):
        best = (-1, -np.inf, 0)  # (fits, covered frames, -error) decides; the unit follows
        chosen = 0
        for length, units in by_length.items():
            motion = trace[frame:frame + length]
            if not len(motion):
                motion = trace[-1:]
            pose = vs.pose_of(motion[:, :6], FORM)
            drafts = vs.decode(memory.values[units], length, FORM, pose, memory.grip_k)[:, :len(motion)]
            errors = np.mean((drafts - motion[None]) ** 2, axis=(1, 2))
            fits = errors <= tau
            if match_gripper:
                fits &= np.all(np.sign(drafts[:, :, 6]) == np.sign(motion[None, :, 6]), axis=1)
            i = int(np.argmin(np.where(fits, errors, np.inf))) if fits.any() else int(np.argmin(errors))
            rank = (int(fits[i]), len(motion) if fits[i] else 0, -float(errors[i]))
            if rank > best:
                best, chosen = rank, int(units[i])
        return chosen
    return select
