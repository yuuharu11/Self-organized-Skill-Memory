"""Offline deterministic G0/G1 replay from memory-pilot rollout archives."""
import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.policies.policy_config import create_trained_policy
from openpi.training import config_moe_skill

ROOT = pathlib.Path(__file__).resolve().parents[4]
STATE = pathlib.Path(__import__("os").environ.get("SKILLNET_STATE_DIR", ROOT / ".skillnet-state"))
CHECKPOINT = pathlib.Path(__import__("os").environ.get("SKILLNET_MEMORY_PILOT_CHECKPOINT", STATE / "checkpoints/pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999"))


def _shuffle_skill_ids(valid_ids, suite_pool, rng):
    """Change suite-supported skill content, not just order of an unordered set."""
    pool = np.unique(np.asarray(suite_pool, np.int32))
    valid = np.asarray(valid_ids, np.int32)
    if not len(pool):
        raise ValueError("manifest has no suite_skill_ids; cannot guarantee a valid skill shuffle")
    alternatives = [pool[pool != skill_id] for skill_id in valid]
    if any(not len(values) for values in alternatives):
        raise ValueError("cannot shuffle a skill: suite pool has no alternative ID")
    shuffled = np.asarray([rng.choice(values) for values in alternatives], np.int32)
    if np.array_equal(np.sort(shuffled), np.sort(valid)) and len(valid):
        # With just two suite IDs, replacing both [a,b] gives [b,a], which
        # leaves cross-attention unchanged. Change its multiset as well.
        shuffled[0] = rng.choice(pool[pool != shuffled[0]])
    return shuffled


def _replay_indices(n_rows, limit):
    return np.arange(n_rows) if limit <= 0 else np.linspace(0, n_rows - 1, min(n_rows, limit), dtype=int)


def _g1_path_metrics(stack, alphas):
    """Measure deviations from the endpoint segment, including degenerate endpoints."""
    flat = np.asarray(stack, dtype=np.float64).reshape(len(alphas), -1)
    direction = flat[-1] - flat[0]
    length_squared = float(np.dot(direction, direction))
    fractions = ((flat - flat[0]) @ direction / length_squared
                 if length_squared > 1e-24 else np.zeros(len(alphas)))
    closest = flat[0] + np.clip(fractions, 0, 1)[:, None] * direction
    linear = flat[0] + np.asarray(alphas)[:, None] * direction
    return {"g1_endpoint_l2": float(np.sqrt(length_squared)),
            "g1_endpoint_projection_by_alpha": fractions.tolist() if length_squared > 1e-24 else None,
            "g1_distance_to_endpoint_segment_by_alpha": np.linalg.norm(flat - closest, axis=1).tolist(),
            "g1_distance_to_linear_action_by_alpha": np.linalg.norm(flat - linear, axis=1).tolist()}


def _observation(npz, i, skill_ids=None, skill_mask_override=None, embed_override=None):
    skills = (np.asarray(npz["skills"][i][npz["skill_mask"][i]], np.int32) - 1) if skill_ids is None else np.asarray(skill_ids, np.int32)
    # LiberoInputs accepts the source annotation values as a Python-list string.
    result = {
        "observation/image": npz["img_base"][i], "observation/wrist_image": npz["img_wrist"][i],
        "observation/state": npz["proprio"][i], "prompt": str(npz["prompt"][i]),
        "skills": str(skills.tolist()),
    }
    if skill_mask_override is not None:
        result["skill_mask_override"] = skill_mask_override
    if embed_override is not None:
        result["skill_embed_override"] = embed_override
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=0, help="evenly spaced observations per episode; 0 means all")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--g0", action="store_true")
    ap.add_argument("--g1", action="store_true")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if not args.g0 and not args.g1:
        args.g0 = args.g1 = True
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    args.checkpoint = args.checkpoint or pathlib.Path(manifest.get("checkpoint", CHECKPOINT))
    if manifest.get("checkpoint") and args.checkpoint.resolve() != pathlib.Path(manifest["checkpoint"]).resolve():
        raise ValueError("replay checkpoint differs from the recording checkpoint")
    config = config_moe_skill.get_config("pi05_libero_moe_skill_4_40")
    config = dataclasses.replace(config, exp_name=args.checkpoint.parent.name)
    policy = create_trained_policy(config, args.checkpoint, repack_transforms=transforms.Group(), sample_kwargs={"num_steps": 10, "capture_auxiliary": True})
    out = STATE / "outputs" / "memory-pilot" / f"analysis-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    records = []
    capture_equivalence_checked = False
    for ep in manifest["episodes"]:
        data = np.load(args.run_dir / "episodes" / f"{ep['episode_id']}.npz")
        for i in _replay_indices(len(data["step"]), args.steps):
            rng = np.random.default_rng(args.seed + i)
            noise = rng.standard_normal((10, config.model.action_dim)).astype(np.float32)
            correct_result = policy.infer(_observation(data, i), noise=noise)
            correct = correct_result["actions"]
            rec = {"episode_id": ep["episode_id"], "step": int(data["step"][i])}
            selected_norm = np.asarray(correct_result["memory_pilot_selected_expert_contribution_norm"], np.float64)
            shared_norm = np.asarray(correct_result["memory_pilot_shared_expert_norm"], np.float64)
            rec["selected_contribution_to_shared_expert_output_norm_ratio_by_layer"] = np.mean(
                selected_norm / np.maximum(shared_norm, 1e-12), axis=-1
            ).tolist()
            if not capture_equivalence_checked:
                policy._sample_kwargs["capture_auxiliary"] = False
                try:
                    plain_action = policy.infer(_observation(data, i), noise=noise)["actions"]
                finally:
                    policy._sample_kwargs["capture_auxiliary"] = True
                rec["capture_disabled_l2_all"] = float(np.linalg.norm(correct - plain_action))
                rec["capture_disabled_max_abs"] = float(np.max(np.abs(correct - plain_action)))
                if not np.array_equal(correct, plain_action):
                    raise ValueError("capture changes policy actions for the same observation/noise")
                capture_equivalence_checked = True
            if args.g0:
                repeated = policy.infer(_observation(data, i), noise=noise)["actions"]
                valid = data["skills"][i][data["skill_mask"][i]].astype(np.int32) - 1
                if len(valid):
                    shuffled = _shuffle_skill_ids(valid, manifest.get("suite_skill_ids", []), rng)
                    alt_result = policy.infer(_observation(data, i, skill_ids=shuffled), noise=noise)
                    masked_result = policy.infer(_observation(data, i, skill_mask_override=np.zeros(max(4, len(valid)), bool)), noise=noise)
                    alt = alt_result["actions"]
                    masked = masked_result["actions"]
                    rec["shuffle_expert_top1_change_by_layer"] = np.mean(
                        correct_result["memory_pilot_router_top1"] != alt_result["memory_pilot_router_top1"], axis=-1
                    ).tolist()
                    rec["masked_expert_top1_change_by_layer"] = np.mean(
                        correct_result["memory_pilot_router_top1"] != masked_result["memory_pilot_router_top1"], axis=-1
                    ).tolist()
                else:
                    alt, masked = correct.copy(), correct.copy()
                noise_result = policy.infer(_observation(data, i), noise=rng.standard_normal(noise.shape).astype(np.float32))
                noise_alt = noise_result["actions"]
                rec["recorded_skill_ids"] = valid.tolist()
                rec["shuffled_skill_ids"] = shuffled.tolist() if len(valid) else []
                rec.update(_distance_metrics(correct, alt, "shuffle")); rec.update(_distance_metrics(correct, masked, "masked")); rec.update(_distance_metrics(correct, noise_alt, "noise"))
                rec.update(_distance_metrics(correct, repeated, "same_noise_repeat"))
            if args.g1:
                processed = policy._input_transform(_observation(data, i))
                tokens = np.asarray(processed["skills"], np.int32)[None, :]
                emb = np.asarray(policy._model.PaliGemma.llm(jnp.asarray(tokens), method="skill_embed"))[0]
                alphas = [0.0, .25, .5, .75, 1.0]
                outputs = []
                valid = data["skills"][i][data["skill_mask"][i]].astype(np.int32) - 1
                if not len(valid):
                    raise ValueError("G1 needs at least one recorded skill")
                first_id = int(valid[0])
                alternatives = [int(s) for s in valid if s != first_id]
                if not alternatives:
                    alternatives = [int(s) for s in manifest.get("suite_skill_ids", []) if s != first_id]
                if not alternatives:
                    raise ValueError("G1 needs two distinct suite-supported skills")
                second_id = alternatives[0]
                rec["g1_skill_ids"] = [first_id, second_id]
                tokens_a = tokens.copy(); tokens_a[0, 0] = first_id + 1
                tokens_b = tokens.copy(); tokens_b[0, 0] = second_id + 1
                emb_a = np.asarray(policy._model.PaliGemma.llm(jnp.asarray(tokens_a), method="skill_embed"))[0, 0]
                emb_b = np.asarray(policy._model.PaliGemma.llm(jnp.asarray(tokens_b), method="skill_embed"))[0, 0]
                single_mask = np.zeros(tokens.shape[1], dtype=bool); single_mask[0] = True
                for alpha in alphas:
                    override = np.zeros_like(emb)
                    override[0] = (1 - alpha) * emb_a + alpha * emb_b
                    outputs.append(policy.infer(_observation(data, i, skill_mask_override=single_mask,
                                                              embed_override=override), noise=noise)["actions"])
                # Compare interpolation against a two-token set attended by the normal mask path.
                set_override = np.zeros_like(emb)
                set_override[0], set_override[1] = emb_a, emb_b
                set_mask = np.zeros(tokens.shape[1], dtype=bool); set_mask[:2] = True
                two_token_set = policy.infer(_observation(data, i, skill_mask_override=set_mask,
                                                          embed_override=set_override), noise=noise)["actions"]
                stack = np.stack(outputs)
                rec.update(_g1_path_metrics(stack, alphas))
                rec["g1_alpha_adjacent_l2"] = np.linalg.norm(np.diff(stack, axis=0), axis=-1).mean(axis=-1).tolist()
                rec["g1_midpoint_endpoint_l2"] = float(np.linalg.norm(stack[2] - stack[0]))
                rec["g1_midpoint_endpoint1_l2"] = float(np.linalg.norm(stack[2] - stack[4]))
                rec["g1_two_token_set_midpoint_l2"] = float(np.linalg.norm(two_token_set - stack[2]))
            records.append(rec)
    (out / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "report.md").write_text(_report(args, manifest, records))
    print(out)


def _distance_metrics(a, b, prefix):
    d = np.asarray(a) - np.asarray(b)
    return {f"{prefix}_l2_all": float(np.linalg.norm(d)),
            f"{prefix}_rms_per_action_value": float(np.sqrt(np.mean(np.square(d)))),
            f"{prefix}_l2_pos": float(np.linalg.norm(d[..., :3])),
            f"{prefix}_l2_rot": float(np.linalg.norm(d[..., 3:6])),
            f"{prefix}_l2_gripper": float(np.linalg.norm(d[..., 6:7]))}


def _report(args, manifest, rows):
    g0 = [r for r in rows if "shuffle_l2_all" in r]
    g1 = [r for r in rows if "g1_midpoint_endpoint_l2" in r]
    lines = ["# Offline memory pilot analysis", "", f"Input: `{args.run_dir}`", f"Checkpoint: `{args.checkpoint}`", f"Observations: {len(rows)}", "",
             "Action distances use the policy output after `Unnormalize` and `LiberoOutputs` (denormalized dataset-action command units, dimensions 0:3 / 3:6 / 6; these are not end-effector pose meters/radians). `*_l2_all` is the Euclidean norm over the full chunk (horizon × 7); `*_rms_per_action_value` is RMS per scalar.", "", "## G0", ""]
    lines.insert(-2, "Expert/norm diagnostics are from the final denoising evaluation only. The expert gate is the router probability, before the block's residual/adaRMS gate.")
    if g0:
        for key in ("shuffle_l2_all", "masked_l2_all", "noise_l2_all", "same_noise_repeat_l2_all", "shuffle_rms_per_action_value", "masked_rms_per_action_value", "noise_rms_per_action_value", "shuffle_l2_pos", "shuffle_l2_rot", "shuffle_l2_gripper"):
            vals = [r[key] for r in g0]
            lines.append(f"- {key}: mean={np.mean(vals):.6g}, std={np.std(vals):.6g}")
        for prefix in ("shuffle", "masked"):
            choices = [r[f"{prefix}_expert_top1_change_by_layer"] for r in g0 if f"{prefix}_expert_top1_change_by_layer" in r]
            if choices:
                means = np.mean(np.asarray(choices), axis=0)
                lines.append(f"- {prefix} expert top-1 change rate by layer (fraction of suffix tokens): " + ", ".join(f"L{i}={v:.4f}" for i, v in enumerate(means)))
        ratios = [r["selected_contribution_to_shared_expert_output_norm_ratio_by_layer"] for r in g0
                  if "selected_contribution_to_shared_expert_output_norm_ratio_by_layer" in r]
        if ratios:
            means = np.mean(np.asarray(ratios), axis=0)
            lines.append("- Gated selected-expert contribution/shared-expert output L2-norm ratio by layer (mean over suffix tokens and observations): " +
                         ", ".join(f"L{i}={v:.4g}" for i, v in enumerate(means)))
    else: lines.append("Not run.")
    lines += ["", "## G1", ""]
    if g1:
        lines.append(f"- Mean midpoint distance to alpha=0: {np.mean([r['g1_midpoint_endpoint_l2'] for r in g1]):.6g}")
        lines.append(f"- Mean midpoint distance to alpha=1: {np.mean([r['g1_midpoint_endpoint1_l2'] for r in g1]):.6g}")
        lines.append(f"- Mean two-token-set distance to alpha=0.5: {np.mean([r['g1_two_token_set_midpoint_l2'] for r in g1]):.6g}")
        lines.append("- Per-alpha adjacent distances are in metrics.json. Interpolation uses one active token; the set baseline attends to two tokens.")
        lines.append("- Endpoint separation, projection and distance to the endpoint segment/linear action are in metrics.json. Distances to endpoints alone do not establish that an output lies between them.")
    else: lines.append("Not run.")
    lines += ["", "Seeded noise and same recorded observations were used. Small smoke-test samples are descriptive only.", ""]
    equivalence = [r for r in rows if "capture_disabled_l2_all" in r]
    if equivalence:
        lines.insert(-1, f"Capture on/off action equivalence (same observation/noise): L2={equivalence[0]['capture_disabled_l2_all']:.6g}, max abs={equivalence[0]['capture_disabled_max_abs']:.6g}.")
    return "\n".join(lines)

if __name__ == "__main__": main()
