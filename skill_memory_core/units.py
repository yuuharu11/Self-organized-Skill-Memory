"""Variable-count label-free M6 unit formation for raw action windows."""
from __future__ import annotations

import numpy as np


class RawFunctional:
    """Pairwise functional metric equal to mean squared error over 70 values."""

    def embedding(self, value):
        x = np.asarray(value, np.float32).reshape(-1)
        if x.shape != (70,):
            raise ValueError("RawFunctional expects active7 × 10 = 70 values")
        return x / np.sqrt(70.)

    def distance(self, left, right):
        d = self.embedding(left) - self.embedding(right)
        return float(d @ d)

    def nearest(self, value, centers):
        if not len(centers):
            return None, float("inf")
        c = np.asarray(centers, np.float32)
        e = self.embedding(value)
        ce = c / np.sqrt(70.)
        d = np.sum((ce - e[None]) ** 2, axis=1)
        i = min(range(len(d)), key=lambda j: (float(d[j]), j))
        return int(i), float(d[i])


def medoid(member_indices, values, metric: RawFunctional | None = None) -> int:
    """Return the real member minimizing mean within-unit distance."""
    ix = np.asarray(member_indices, np.int64)
    x = np.asarray(values, np.float32).reshape(len(values), -1)
    if not len(ix):
        raise ValueError("cannot choose medoid of an empty unit")
    if x.shape[1] != 70:
        raise ValueError("raw medoid expects active7×10 vectors")
    v = x[ix] / np.sqrt(70.)
    norm = np.einsum("ij,ij->i", v, v)
    d = np.maximum(norm[:, None] + norm[None, :] - 2 * v @ v.T, 0.)
    return int(ix[np.argmin(d.mean(axis=1))])


def m6_fit(values, row_ids, tasks, episodes, *, kappa: float = .1, seed: int = 20261104,
           min_support: int = 10, min_episodes: int = 3,
           pending_batch: int = 200, pending_max_age: int = 1000,
           epsilon_override: float | None = None):
    """Mirror seeded M6 assign/pending/cluster/register/merge lifecycle.

    The returned unit count is data-dependent. `row_units` aligns to `values`
    input order and uses -1 for unassigned rows. Member lists use local indices.
    """
    x = np.asarray(values, np.float32).reshape(len(values), -1)
    ids, ts, es = np.asarray(row_ids, np.int64), np.asarray(tasks, np.int32), np.asarray(episodes, np.int64)
    if x.shape[1] != 70 or not (len(x) == len(ids) == len(ts) == len(es)) or not len(x):
        raise ValueError("M6 inputs must align, be nonempty, and contain 70-D windows")
    rng = np.random.default_rng(int(seed) + 6046)
    pair = rng.integers(0, len(x), size=(64, 2))
    metric = RawFunctional()
    median = float(np.median([metric.distance(x[a], x[b]) for a, b in pair]))
    eps = float(epsilon_override if epsilon_override is not None else max(kappa * median, 1e-12))
    centers, supports, unit_eps, members = [], [], [], []
    row_units = np.full(len(x), -1, np.int32)
    pending, pending_age = [], {}
    pending_since = next_pending = expired = 0

    def assign_pending(force=False):
        nonlocal pending, pending_since, expired
        if not pending or (not force and pending_since < pending_batch):
            return
        pending_since = 0
        clusters = []
        for item in pending:
            if not clusters:
                clusters.append({"values": [item], "center": item["value"].copy(),
                                 "eps": {item["episode_key"]}})
                continue
            j, dist = metric.nearest(item["value"], [c["center"] for c in clusters])
            if dist <= eps:
                c = clusters[j]
                c["values"].append(item)
                c["center"] = np.mean([v["value"] for v in c["values"]], axis=0).astype(np.float32)
                c["eps"].add(item["episode_key"])
            else:
                clusters.append({"values": [item], "center": item["value"].copy(),
                                 "eps": {item["episode_key"]}})
        registered = {}
        for c in clusters:
            vals = c["values"]
            if len(vals) < min_support or len(c["eps"]) < min_episodes:
                continue
            j, dist = metric.nearest(c["center"], centers)
            if j is not None and dist <= eps:
                old, count = supports[j], len(vals)
                centers[j] = ((centers[j] * old + c["center"] * count) / (old + count)).astype(np.float32)
                supports[j] += count
                unit_eps[j].update(c["eps"])
                members[j].extend(v["local"] for v in vals)
                uid = j
            else:
                uid = len(centers)
                centers.append(c["center"].copy())
                supports.append(len(vals))
                unit_eps.append(set(c["eps"]))
                members.append([v["local"] for v in vals])
            for item in vals:
                registered[item["id"]] = uid
        keep = []
        for item in pending:
            pid = item["id"]
            if pid in registered:
                row_units[item["local"]] = registered[pid]
                pending_age.pop(pid, None)
            else:
                pending_age[pid] = pending_age.get(pid, 0) + pending_batch
                if pending_age[pid] >= pending_max_age:
                    pending_age.pop(pid, None)
                    expired += 1
                else:
                    keep.append(item)
        pending = keep

    for local in rng.permutation(len(x)):
        val = x[local]
        j, dist = metric.nearest(val, centers)
        epkey = (int(ts[local]), int(es[local]))
        if j is not None and dist <= eps:
            old = supports[j]
            centers[j] = ((centers[j] * old + val) / (old + 1)).astype(np.float32)
            supports[j] += 1
            unit_eps[j].add(epkey)
            members[j].append(int(local))
            row_units[local] = j
        else:
            pending.append({"id": next_pending, "local": int(local), "value": val.copy(), "episode_key": epkey})
            pending_age[next_pending] = 0
            next_pending += 1
            pending_since += 1
        assign_pending()
    assign_pending(force=True)
    return {"row_units": row_units, "centers": np.asarray(centers, np.float32).reshape(-1, 70),
            "members_local": [np.asarray(m, np.int64) for m in members],
            "support": np.asarray(supports, np.int32), "episode_sets": unit_eps,
            "functional_median": median, "epsilon": eps,
            "pending_local": np.asarray([v["local"] for v in pending], np.int64),
            "expired_pending": int(expired), "registered": len(centers), "seed": int(seed)}
