"""S1 DMP-to-LIBERO-action adapter used by S3 training and inference."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def dmp_pose_to_normalized_actions(
    trajectory,
    y0,
    gripper,
    *,
    q01,
    q99,
    position_m_per_command: float = 0.05,
    rotation_rad_per_command: float = 0.5,
    action_horizon: int = 10,
):
    """Convert a DMP absolute EEF pose rollout into one normalized 10x7 draft.

    ``trajectory`` and ``y0`` are physical EEF pose coordinates [xyz meters,
    axis-angle radians]. LIBERO OSC controller commands are represented in raw
    dimensionless action units; the scale defaults here match the project
    offline motion conversion (0.05m / unit and 0.5rad / unit) and remain
    overrideable until S1's DMP coordinate metadata is finalized. Quantile
    normalization is the same affine transform as ``transforms.Normalize``.
    The gripper command is copied from the chosen primitive; candidates are
    never averaged.
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32).reshape(-1)[:7]
    q99 = np.asarray(q99, dtype=np.float32).reshape(-1)[:7]
    if trajectory.ndim == 2:
        trajectory = trajectory[None]
    if y0.ndim == 1:
        y0 = np.broadcast_to(y0[None], (trajectory.shape[0], y0.shape[0]))
    if trajectory.ndim != 3 or trajectory.shape[1:] != (action_horizon, 6):
        raise ValueError(f"trajectory must be [B,{action_horizon},6], got {trajectory.shape}")
    if y0.shape != (trajectory.shape[0], 6):
        raise ValueError(f"y0 must be [B,6], got {y0.shape}")
    if q01.shape != (7,) or q99.shape != (7,) or np.any(q99 - q01 < 1e-6):
        raise ValueError("q01/q99 must contain seven valid action quantiles")
    if position_m_per_command <= 0 or rotation_rad_per_command <= 0:
        raise ValueError("controller scales must be positive")

    previous = np.concatenate([y0[:, None], trajectory[:, :-1]], axis=1)
    delta = trajectory - previous
    # LIBERO OSC_POSE applies the commanded axis-angle rotation on the left of
    # the current world orientation (R_goal = R_delta @ R_current). Subtracting
    # two absolute rotvecs is not the same operation for non-commuting poses.
    old_rot = Rotation.from_rotvec(previous[..., 3:].reshape(-1, 3))
    new_rot = Rotation.from_rotvec(trajectory[..., 3:].reshape(-1, 3))
    world_delta_rot = (new_rot * old_rot.inv()).as_rotvec().reshape(trajectory.shape[0], action_horizon, 3)
    delta[..., 3:] = world_delta_rot
    raw = np.empty((trajectory.shape[0], action_horizon, 7), np.float32)
    raw[..., :3] = delta[..., :3] / position_m_per_command
    raw[..., 3:6] = delta[..., 3:6] / rotation_rad_per_command
    grip = np.asarray(gripper, np.float32).reshape(-1)
    if grip.size == 1:
        grip = np.repeat(grip, trajectory.shape[0])
    if grip.shape != (trajectory.shape[0],):
        raise ValueError("gripper must be a scalar or [B]")
    raw[..., 6] = grip[:, None]
    normalized = (raw - q01[None, None]) / (q99 - q01 + 1e-6)[None, None] * 2.0 - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def synthetic_dmp_rollout(weights, y0, goal, tau, *, n_steps=10):
    """Tiny deterministic DMP-like expansion for interface/synthetic checks.

    The production server imports S1's ``dmp.py`` after INTERFACE_READY. This
    smooth basis expansion is test-only and intentionally not a learned model.
    """
    weights = np.asarray(weights, np.float32)
    y0, goal = np.asarray(y0, np.float32), np.asarray(goal, np.float32)
    if weights.ndim != 2 or weights.shape[0] != 6 or y0.shape != (6,) or goal.shape != (6,):
        raise ValueError("expected weights [6,N], y0 [6], goal [6]")
    tau = max(float(tau), 1e-4)
    phase = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    centers = np.linspace(0.0, 1.0, weights.shape[1], dtype=np.float32)
    widths = max(weights.shape[1] ** 1.5, 1.0)
    psi = np.exp(-widths * (phase[:, None] - centers[None]) ** 2)
    basis = psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-8)
    shape = basis @ weights.T / tau
    linear = y0[None] + phase[:, None] * (goal - y0)[None]
    return linear + (phase * (1.0 - phase))[:, None] * shape


def rollout_from_s1(dmp_module, library, prim_id, y0, goal, tau, *, n_steps=10, phase_s=0.0):
    """Invoke S1's public ``rollout(prim_id, y0, g, tau, n_steps)`` API.

    S1's ``shapes`` array is a normalized trajectory dictionary [K,6,N], not
    the DMP basis-weight matrix returned by ``fit``. Passing ``shapes[k]`` as
    the first positional argument would therefore misinterpret a library shape
    as weights. Keep the complete library attached and pass the integer ID.
    """
    shapes = np.asarray(library["shapes"], np.float32)
    prim_id = int(prim_id)
    if not 0 <= prim_id < len(shapes):
        raise ValueError(f"prim_id {prim_id} outside S1 dictionary of size {len(shapes)}")
    configure = getattr(dmp_module, "configure_library", None)
    if configure is not None:
        configure_library = {key: np.asarray(value) for key, value in library.items()}
        configure(configure_library)
    result = dmp_module.rollout(prim_id, y0, goal, tau, n_steps=n_steps, library=library, phase_s=phase_s)
    result = np.asarray(result, np.float32)
    if result.shape != (n_steps, 6):
        raise ValueError(f"S1 rollout must return [{n_steps},6], got {result.shape}")
    return result


def straight_minimum_jerk_shape(n_points: int, *, dtype=np.float32):
    """Canonical straight-to-goal shape used only as the shape ablation.

    The S1 library stores canonical normalized trajectories [6,N], whose
    endpoint is [1,0,0,1,0,0]. This returns that same representation and keeps
    g, tau, phase, and the selected primitive's gripper command untouched.
    """
    if n_points < 2:
        raise ValueError("straight primitive needs at least two points")
    u = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    phase = 10 * u**3 - 15 * u**4 + 6 * u**5
    endpoint = np.asarray([1, 0, 0, 1, 0, 0], np.float64)
    return (endpoint[:, None] * phase[None, :]).astype(dtype)


def build_dmp_draft(
    dmp_module,
    library,
    route,
    y0,
    *,
    q01,
    q99,
    position_m_per_command=0.05,
    rotation_rad_per_command=0.5,
    action_horizon=10,
    shape_override=None,
):
    """Expand one hard route into a normalized [B,H,7] flow-bridge draft.

    ``route`` is a DmpRouteState-like object with batched ``prim_id``, ``goal``,
    ``tau`` and ``phase_s`` arrays. These continuous parameters are retained
    for both rollout and model conditioning.
    """
    prim_ids = np.asarray(route.prim_id).reshape(-1)
    goals = np.asarray(route.goal, np.float32).reshape((-1, 6))
    taus = np.asarray(route.tau, np.float32).reshape(-1)
    y0s = np.asarray(y0, np.float32).reshape((-1, 6))
    if not (len(prim_ids) == len(goals) == len(taus) == len(y0s)):
        raise ValueError("route and y0 batch dimensions differ")
    trajectories = []
    grips = np.asarray(library["grip"]).reshape(-1)
    override = None if shape_override is None else np.asarray(shape_override, np.float32)
    if override is not None and override.shape[0] == 1 and len(prim_ids) > 1:
        override = np.repeat(override, len(prim_ids), axis=0)
    if override is not None and override.shape[0] != len(prim_ids):
        raise ValueError("shape_override must have one [6,N] shape per route")
    for i, (pid, goal, tau, start) in enumerate(zip(prim_ids, goals, taus, y0s, strict=True)):
        if override is None:
            traj = rollout_from_s1(
                dmp_module, library, int(pid), start, goal, float(tau), n_steps=action_horizon,
                phase_s=float(np.asarray(route.phase_s).reshape(-1)[i]),
            )
        else:
            # S1's shape arrays are normalized paths, while rollout also
            # accepts fitted DMP weights. Refit the supplied path in the same
            # public basis, then expand with the unchanged g/tau.
            # A canonical endpoint is +x even when the corresponding world
            # displacement is zero. S1's rollout deliberately uses a small
            # fallback scale for zero-goal loop shapes; for a straight-shape
            # ablation that fallback would create a spurious 1 cm / 0.1 rad
            # motion. Suppress the zero-goal subspaces before fitting.
            shape = override[i].copy()
            if np.linalg.norm(goal[:3]) < 1e-8:
                shape[:3] = 0.0
            if np.linalg.norm(goal[3:]) < 1e-8:
                shape[3:] = 0.0
            weights, _, _ = dmp_module.fit(shape.T)
            traj = np.asarray(dmp_module.rollout(
                weights, start, goal, float(tau), n_steps=action_horizon,
                library=library, phase_s=float(np.asarray(route.phase_s).reshape(-1)[i]),
            ), np.float32)
            # S1's canonical rollout has a fixed +x endpoint in every 3D
            # subspace. Its small fallback scale for a zero world goal is
            # useful for learned loop primitives, but a straight ablation must
            # remain stationary in a subspace with no requested displacement.
            if np.linalg.norm(goal[:3]) < 1e-8:
                traj[:, :3] = start[None, :3]
            if np.linalg.norm(goal[3:]) < 1e-8:
                traj[:, 3:] = start[None, 3:]
        trajectories.append(traj)
    return dmp_pose_to_normalized_actions(
        np.stack(trajectories), y0s, grips[prim_ids.astype(np.int64)],
        q01=q01, q99=q99,
        position_m_per_command=position_m_per_command,
        rotation_rad_per_command=rotation_rad_per_command,
        action_horizon=action_horizon,
    )
