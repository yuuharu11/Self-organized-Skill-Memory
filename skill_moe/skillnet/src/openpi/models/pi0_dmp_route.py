"""Primitive-routed bridge path for the LIBERO-90 SkillNet π0.5 model.

This is an opt-in model class: the released Pi0/SkillNet code paths are not
modified. S1 supplies a variable-size, unlabeled DMP dictionary; S2 labels are
used only as router supervision during training. At inference, ``route_from_``
``observation`` predicts a hard top-1 primitive and its continuous parameters.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma_moe_skill as _gemma
from openpi.models import model as _model
from openpi.models.pi0_moe_skill import Pi0 as _SkillNetPi0
from openpi.models.pi0_moe_skill import make_attn_mask
from openpi.shared import array_typing as at


class DmpRoutePrediction(NamedTuple):
    """Content-pointer scores and continuous route outputs."""

    prim_logits: jax.Array  # [B,K], q(prefix, history) dot E(shape[k])
    goal: jax.Array  # [B,6]
    log_tau: jax.Array  # [B]
    phase_logit: jax.Array  # [B]
    query: jax.Array  # [B,D]
    shape_embeddings: jax.Array  # [K,D]


class DmpRouteState(NamedTuple):
    """Hard primitive route and its decoded continuous parameters."""

    prim_id: jax.Array  # [B], int32
    shape_embedding: jax.Array  # [B,D], E(w_selected)
    goal: jax.Array  # [B,6]
    tau: jax.Array  # [B], positive
    phase_s: jax.Array  # [B], in [0,1]


class DmpRouterHead(nnx.Module):
    """Query encoder q(prefix, ordered self-state history); output size is K-free."""

    def __init__(self, *, prefix_width: int, state_width: int, state_history_len: int,
                 hidden_dim: int, shape_width: int, rngs: nnx.Rngs):
        self.state_history_len = int(state_history_len)
        self.prefix_norm = nnx.LayerNorm(prefix_width, rngs=rngs)
        self.prefix_fc = nnx.Linear(prefix_width, hidden_dim, rngs=rngs)
        self.state_norm = nnx.LayerNorm(state_width * state_history_len, rngs=rngs)
        self.state_fc = nnx.Linear(state_width * state_history_len, hidden_dim, rngs=rngs)
        self.fuse = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.query = nnx.Linear(hidden_dim, shape_width, rngs=rngs)
        self.goal = nnx.Linear(hidden_dim, 6, rngs=rngs)
        self.log_tau = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.phase = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, prefix_hidden: jax.Array, prefix_mask: jax.Array,
                 state_history: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if prefix_hidden.ndim != 3 or prefix_mask.shape != prefix_hidden.shape[:2]:
            raise ValueError("prefix_hidden/mask must have [B,T,D] and [B,T] shapes")
        if state_history.ndim != 3 or state_history.shape[0] != prefix_hidden.shape[0]:
            raise ValueError("state_history must have [B,L,S] shape")
        if state_history.shape[1] != self.state_history_len:
            raise ValueError(f"state history length must be {self.state_history_len}")
        weights = prefix_mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(prefix_hidden.astype(jnp.float32) * weights, axis=1)
        pooled /= jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        p = nnx.gelu(self.prefix_fc(self.prefix_norm(pooled)))
        s = state_history.astype(jnp.float32).reshape((state_history.shape[0], -1))
        s = nnx.gelu(self.state_fc(self.state_norm(s)))
        h = nnx.gelu(self.fuse(jnp.concatenate([p, s], axis=-1)))
        query = self.query(h).astype(jnp.float32)
        query /= jnp.maximum(jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
        return query, self.goal(h).astype(jnp.float32), self.log_tau(h)[..., 0].astype(jnp.float32), self.phase(h)[..., 0].astype(jnp.float32)


class DmpShapeEncoder(nnx.Module):
    """K-independent E(w); adding a dictionary row never adds model parameters."""

    def __init__(self, *, basis_n: int, hidden_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.basis_n = int(basis_n)
        self.norm = nnx.LayerNorm(6 * self.basis_n, rngs=rngs)
        self.fc1 = nnx.Linear(6 * self.basis_n, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, shapes: jax.Array) -> jax.Array:
        shapes = jnp.asarray(shapes, jnp.float32)
        if shapes.ndim != 3 or shapes.shape[1:] != (6, self.basis_n):
            raise ValueError(f"shape library must be [K,6,{self.basis_n}], got {shapes.shape}")
        flat = shapes.reshape((shapes.shape[0], -1))
        embedding = self.fc2(nnx.gelu(self.fc1(self.norm(flat)))).astype(jnp.float32)
        return embedding / jnp.maximum(jnp.linalg.norm(embedding, axis=-1, keepdims=True), 1e-6)


class DmpConditionEncoder(nnx.Module):
    """Trainable primitive embedding plus (g, log τ, phase) projections.

    The router embedding replaces task-annotation skill IDs. The independent
    adaRMS projection is added to the action expert's time condition. These
    can be ablated separately via ``injection_mode``.
    """

    def __init__(
        self,
        *,
        router_width: int,
        action_width: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.router_width = int(router_width)
        self.cont_in = nnx.Linear(8, hidden_dim, rngs=rngs)
        self.cont_out = nnx.Linear(hidden_dim, router_width, rngs=rngs)
        self.ada_in = nnx.Linear(router_width + 8, hidden_dim, rngs=rngs)
        self.ada_out = nnx.Linear(hidden_dim, action_width, rngs=rngs)

    @staticmethod
    def _continuous(goal, tau, phase_s):
        goal = jnp.asarray(goal, jnp.float32)
        tau = jnp.asarray(tau, jnp.float32).reshape((-1, 1))
        phase_s = jnp.asarray(phase_s, jnp.float32).reshape((-1, 1))
        if goal.ndim != 2 or goal.shape[-1] != 6:
            raise ValueError(f"goal must be [B,6], got {goal.shape}")
        return jnp.concatenate([goal, jnp.log(jnp.maximum(tau, 1e-4)), phase_s], axis=-1)

    def _components(self, route: DmpRouteState):
        primitive = jnp.asarray(route.shape_embedding, jnp.float32)
        if primitive.ndim != 2 or primitive.shape[0] != route.goal.shape[0] or primitive.shape[1] != self.router_width:
            raise ValueError(f"shape embedding must be [B,{self.router_width}], got {primitive.shape}")
        continuous = self._continuous(route.goal, route.tau, route.phase_s)
        projected = self.cont_out(nnx.gelu(self.cont_in(continuous)))
        return primitive, continuous, primitive + projected

    def router_tokens(self, route: DmpRouteState) -> jax.Array:
        _, _, token = self._components(route)
        return token[:, None, :]

    def adarms_delta(self, route: DmpRouteState) -> jax.Array:
        primitive, continuous, _ = self._components(route)
        joined = jnp.concatenate([primitive, continuous], axis=-1)
        return self.ada_out(nnx.gelu(self.ada_in(joined))).astype(jnp.float32)


def score_shape_dictionary(query: jax.Array, shape_embeddings: jax.Array) -> jax.Array:
    """Content-pointer scores for a runtime dictionary of any nonzero size."""
    query = jnp.asarray(query, jnp.float32)
    shape_embeddings = jnp.asarray(shape_embeddings, jnp.float32)
    if query.ndim != 2 or shape_embeddings.ndim != 2 or query.shape[-1] != shape_embeddings.shape[-1]:
        raise ValueError("query [B,D] and shape_embeddings [K,D] must share D")
    if shape_embeddings.shape[0] < 1:
        raise ValueError("primitive dictionary cannot be empty")
    return query @ shape_embeddings.T


def decode_route(prediction: DmpRoutePrediction, *, min_tau: float = 0.05) -> DmpRouteState:
    """Apply hard top-1 routing and bounded physical parameter transforms."""
    prim_id = jnp.argmax(prediction.prim_logits, axis=-1).astype(jnp.int32)
    shape_embedding = prediction.shape_embeddings[prim_id]
    # The route loss regresses this head to log(target_tau), so decode with the
    # matching exponential transform. Softplus(log_tau) is not an inverse of
    # that target and systematically changes the demonstrated duration.
    tau = jnp.maximum(jnp.exp(prediction.log_tau), min_tau)
    return DmpRouteState(
        prim_id,
        shape_embedding,
        prediction.goal,
        tau,
        jax.nn.sigmoid(prediction.phase_logit),
    )


def shuffle_route_primitives(route: DmpRouteState, rng: at.KeyArrayLike, shape_embeddings: jax.Array) -> DmpRouteState:
    """Deterministically keyed primitive-ID permutation for ablation runs.

    Geometry (g, τ, phase) is held fixed; the caller should expand a new
    single-primitive draft from the shuffled ID. The permutation is fixed per
    call and does not inspect annotations or targets.
    """
    shape_embeddings = jnp.asarray(shape_embeddings, jnp.float32)
    num_primitives = shape_embeddings.shape[0]
    if num_primitives < 1:
        raise ValueError("primitive shuffle requires a nonempty dictionary")
    permutation = jax.random.permutation(rng, num_primitives)
    ids = jnp.mod(jnp.asarray(route.prim_id, jnp.int32), num_primitives)
    shuffled_ids = permutation[ids]
    return route._replace(prim_id=shuffled_ids, shape_embedding=shape_embeddings[shuffled_ids])


def bridge_source(draft: jax.Array, noise: jax.Array, *, t0: float = 0.4) -> jax.Array:
    """Create a one-draft bridge source in the SkillNet 10x32 input layout.

    Active coordinates 0:7 use x_t0=(1-t0)A+t0ε. Coordinates 7:14 carry A
    as fixed context on every denoising step; 14:32 are zero. A is never mixed
    with another primitive or demo.
    """
    draft = jnp.asarray(draft, jnp.float32)
    noise = jnp.asarray(noise, jnp.float32)
    if draft.ndim != 3 or draft.shape[-1] != 7:
        raise ValueError(f"draft must have [B,H,7], got {draft.shape}")
    if noise.ndim != 3 or noise.shape[:2] != draft.shape[:2] or noise.shape[-1] < 7:
        raise ValueError("noise must have [B,H,D>=7] matching the draft")
    if not 0.0 <= t0 <= 1.0:
        raise ValueError("t0 must be in [0,1]")
    x = jnp.zeros((draft.shape[0], draft.shape[1], 32), jnp.float32)
    x = x.at[..., :7].set((1.0 - t0) * draft + t0 * noise[..., :7])
    x = x.at[..., 7:14].set(draft)
    return x


class Pi0DmpRoute(_SkillNetPi0):
    """SkillNet-90 backbone with a learned DMP router and bridge sampler."""

    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.dmp_config = config
        if config.enable_primitive_router:
            prefix_width = _gemma.get_config(config.paligemma_variant).width
            action_width = _gemma.get_config(config.action_expert_variant).width
            router_width = config.skill_embed_dim + (4 if config.positional_encoding else 0)
            self.dmp_router = DmpRouterHead(
                prefix_width=prefix_width,
                state_width=config.action_dim,
                state_history_len=config.state_history_len,
                hidden_dim=config.router_hidden_dim,
                shape_width=router_width,
                rngs=rngs,
            )
            self.dmp_shape_encoder = DmpShapeEncoder(
                basis_n=config.shape_basis_n,
                hidden_dim=config.router_hidden_dim,
                output_dim=router_width,
                rngs=rngs,
            )
            self.dmp_condition = DmpConditionEncoder(
                router_width=router_width,
                action_width=action_width,
                hidden_dim=config.router_hidden_dim,
                rngs=rngs,
            )
        else:
            self.dmp_router = None
            self.dmp_shape_encoder = None
            self.dmp_condition = None

    def _null_skill(self, batch: int, dtype):
        width = self.dmp_config.skill_embed_dim + (4 if self.positional_encoding else 0)
        # An explicit all-masked token is the true unconditioned path. An
        # all-zero but valid token still produces a learned Dense bias in the
        # MoE cross-attention and is not equivalent to no primitive signal.
        return jnp.zeros((batch, 1, width), dtype=dtype), jnp.zeros((batch, 1), jnp.bool_)

    def encode_primitive_shapes(self, primitive_shapes: jax.Array) -> jax.Array:
        if self.dmp_shape_encoder is None:
            raise ValueError("this model was created without the S3 shape encoder")
        return self.dmp_shape_encoder(primitive_shapes)

    def route_from_observation(
        self,
        observation: _model.Observation,
        primitive_shapes: jax.Array,
        state_history: jax.Array | None = None,
    ) -> DmpRoutePrediction:
        """Score a runtime shape dictionary from prefix and self-state history."""
        if self.dmp_router is None:
            raise ValueError("this model was created without an S3 primitive router")
        observation = _model.preprocess_observation(None, observation, train=False)
        primitive_shapes = jnp.asarray(primitive_shapes, jnp.float32)
        shape_embeddings = self.encode_primitive_shapes(primitive_shapes)
        if state_history is None:
            # A single live observation has no prior history unless the serving
            # wrapper supplies one. Replication is a deterministic cold-start.
            state_history = jnp.repeat(
                observation.state[:, None, :], self.dmp_config.state_history_len, axis=1
            )
        else:
            state_history = jnp.asarray(state_history, jnp.float32)
        if state_history.shape != (
            observation.state.shape[0], self.dmp_config.state_history_len, observation.state.shape[-1]
        ):
            raise ValueError("state_history must be [B,state_history_len,state_width]")
        prefix, mask, ar = self.embed_prefix(observation)
        attn = make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=-1) - 1
        # No task/skill annotations enter this pass; a null router token keeps
        # the base MoE call well-defined while the route is being predicted.
        null_skill, null_mask = self._null_skill(prefix.shape[0], prefix.dtype)
        (hidden, _), _, _ = self.PaliGemma.llm(
            [prefix, None], skill_embeds=null_skill, skill_mask=null_mask,
            mask=attn, positions=pos, adarms_cond=[None, None],
        )
        # PaliGemma is frozen. Do not retain its activations for the small
        # supervised router head.
        query, goal, log_tau, phase_logit = self.dmp_router(
            jax.lax.stop_gradient(hidden), mask, state_history
        )
        return DmpRoutePrediction(
            score_shape_dictionary(query, shape_embeddings), goal, log_tau, phase_logit,
            query, shape_embeddings,
        )

    def predict_route(self, observation, primitive_shapes, state_history=None) -> DmpRouteState:
        return decode_route(self.route_from_observation(observation, primitive_shapes, state_history))

    def _router_condition(self, route: DmpRouteState, injection_mode: str):
        if injection_mode not in ("both", "router_only", "adarms_only", "none"):
            raise ValueError(f"unsupported injection mode {injection_mode!r}")
        batch = route.goal.shape[0]
        if injection_mode == "none":
            skill, skill_mask = self._null_skill(batch, route.goal.dtype)
            return skill, skill_mask, None
        if self.dmp_condition is None:
            raise ValueError("primitive conditioning requested without a dictionary")
        if injection_mode in ("both", "router_only"):
            skill = self.dmp_condition.router_tokens(route)
            skill_mask = jnp.ones(skill.shape[:2], jnp.bool_)
        else:
            skill, skill_mask = self._null_skill(batch, route.goal.dtype)
        delta = (
            self.dmp_condition.adarms_delta(route)
            if injection_mode in ("both", "adarms_only")
            else None
        )
        return skill, skill_mask, delta

    def _bridge_prediction(self, observation, x, time, route, *, injection_mode):
        suffix, sm, sa, adarms = super().embed_suffix(observation, x, time)
        skill, skill_mask, delta = self._router_condition(route, injection_mode)
        if delta is not None:
            adarms = adarms + delta.astype(adarms.dtype)
        prefix, pm, pa = self.embed_prefix(observation)
        all_mask = jnp.concatenate([pm, sm], axis=1)
        all_ar = jnp.concatenate([pa, jnp.broadcast_to(sa, sm.shape)], axis=1)
        attn = make_attn_mask(all_mask, all_ar)
        positions = jnp.cumsum(all_mask, axis=-1) - 1
        (_, out), _, balance = self.PaliGemma.llm(
            [prefix, suffix], skill_embeds=skill, skill_mask=skill_mask,
            mask=attn, positions=positions, adarms_cond=[None, adarms],
        )
        return self.action_out_proj(out[:, -self.action_horizon :]), balance

    def sample_actions_s3(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        draft: jax.Array | None = None,
        noise: jax.Array | None = None,
        route: DmpRouteState | None = None,
        primitive_shapes: jax.Array | None = None,
        state_history: jax.Array | None = None,
        t0: float | None = None,
        num_steps: int = 2,
        injection_mode: str | None = None,
    ) -> _model.Actions:
        """Integrate one DMP-derived draft; the router is predicted if omitted.

        ``draft`` is already converted/quantile-normalized by the S1 adapter and
        has [B,H,7]. The generic library/pose/action conversion stays outside
        this model so the exact S1 coordinate convention is swappable.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        mode = self.dmp_config.injection_mode if injection_mode is None else injection_mode
        if route is None:
            if self.dmp_router is not None:
                if primitive_shapes is None:
                    raise ValueError("runtime primitive_shapes are required for pointer routing")
                route = self.predict_route(observation, primitive_shapes, state_history)
            elif mode == "none":
                route = DmpRouteState(
                    jnp.zeros((batch,), jnp.int32),
                    jnp.zeros((batch, self.dmp_config.skill_embed_dim + (4 if self.positional_encoding else 0)), jnp.float32),
                    jnp.zeros((batch, 6), jnp.float32),
                    jnp.ones((batch,), jnp.float32),
                    jnp.zeros((batch,), jnp.float32),
                )
            else:
                raise ValueError("primitive-conditioned sampling requires an S1 dictionary")
        if draft is None:
            if mode != "none":
                raise ValueError("primitive-conditioned sampling requires one DMP draft")
            draft = jnp.zeros((batch, self.action_horizon, 7), jnp.float32)
        draft = jnp.asarray(draft, jnp.float32)
        expected = (batch, self.action_horizon, 7)
        if draft.shape != expected:
            raise ValueError(f"draft shape {draft.shape} != {expected}")
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
        else:
            noise = jnp.asarray(noise, jnp.float32)
        start_t = float(self.dmp_config.bridge_t0 if t0 is None else t0)
        if not 0.0 < start_t <= 1.0:
            raise ValueError("bridge t0 must be in (0,1]")
        x = bridge_source(draft, noise, t0=start_t)
        dt = start_t / num_steps

        def body(i, current):
            # Keep the DMP draft as fixed context rather than integrating it.
            current = current.at[..., 7:14].set(draft)
            current = current.at[..., 14:32].set(0.0)
            time = jnp.full((batch,), start_t * (1.0 - i / num_steps), jnp.float32)
            velocity, _ = self._bridge_prediction(
                observation, current, time, route, injection_mode=mode
            )
            return current.at[..., :7].add(-dt * velocity[..., :7])

        result = jax.lax.fori_loop(0, num_steps, body, x)
        # Only the seven LIBERO action dimensions are returned to the policy.
        return jnp.concatenate([result[..., :7], jnp.zeros_like(result[..., 7:])], axis=-1)

    def compute_s3_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        dmp_draft: jax.Array | None = None,
        primitive_shapes: jax.Array | None = None,
        state_history: jax.Array | None = None,
        target_prim_id: jax.Array | None = None,
        target_goal: jax.Array | None = None,
        target_tau: jax.Array | None = None,
        target_phase_s: jax.Array | None = None,
        use_primitives: bool = True,
        train: bool = True,
    ):
        """Flow loss plus S1 self-decomposition route supervision.

        For S3, the bridge uses t0=.4, B=(1-t0)A_dmp+t0ε,
        X_s=(1-s)A_gt+sB and velocity (B-A_gt)/t0. G0 uses the ordinary
        flow-matching path with no primitive/draft/injection.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch = actions.shape[0]
        active = jnp.asarray(actions[..., :7], jnp.float32)
        noise = jax.random.normal(noise_rng, actions.shape, dtype=jnp.float32)
        route_loss = jnp.asarray(0.0, jnp.float32)
        zero_route = DmpRouteState(
            jnp.zeros((batch,), jnp.int32),
            jnp.zeros((batch, self.dmp_config.skill_embed_dim + (4 if self.positional_encoding else 0)), jnp.float32),
            jnp.zeros((batch, 6), jnp.float32),
            jnp.ones((batch,), jnp.float32), jnp.zeros((batch,), jnp.float32),
        )
        if use_primitives:
            if any(v is None for v in (dmp_draft, primitive_shapes, target_prim_id, target_goal, target_tau, target_phase_s)):
                raise ValueError("S3 loss requires DMP draft and the self-decomposition labels")
            prediction = self.route_from_observation(observation, primitive_shapes, state_history)
            target_prim_id = jnp.asarray(target_prim_id, jnp.int32).reshape((batch,))
            target_goal = jnp.asarray(target_goal, jnp.float32).reshape((batch, 6))
            target_tau = jnp.asarray(target_tau, jnp.float32).reshape((batch,))
            target_phase_s = jnp.asarray(target_phase_s, jnp.float32).reshape((batch,))
            ce = jnp.mean(optax_softmax_cross_entropy(prediction.prim_logits, target_prim_id))
            reg_g = jnp.mean(jnp.square(prediction.goal - target_goal))
            reg_tau = jnp.mean(jnp.square(prediction.log_tau - jnp.log(jnp.maximum(target_tau, 1e-4))))
            reg_phase = jnp.mean(jnp.square(jax.nn.sigmoid(prediction.phase_logit) - target_phase_s))
            route_loss = self.dmp_config.router_ce_weight * ce + self.dmp_config.router_reg_weight * (
                reg_g + reg_tau + reg_phase
            )
            selected_shape = prediction.shape_embeddings[target_prim_id]
            route_for_flow = DmpRouteState(target_prim_id, selected_shape, target_goal, target_tau, target_phase_s)
            t0 = float(self.dmp_config.bridge_t0)
            sample_s = jax.random.uniform(time_rng, (batch,), minval=0.0, maxval=1.0)
            endpoint = (1.0 - t0) * jnp.asarray(dmp_draft, jnp.float32) + t0 * noise[..., :7]
            noisy_active = (1.0 - sample_s[:, None, None]) * active + sample_s[:, None, None] * endpoint
            time = t0 * sample_s
            target_velocity = (endpoint - active) / t0
        else:
            # Matched G0: use the same t0 bridge and noise distribution with
            # A=0, and turn off both primitive injection paths.
            route_for_flow = zero_route
            t0 = float(self.dmp_config.bridge_t0)
            sample_s = jax.random.uniform(time_rng, (batch,), minval=0.0, maxval=1.0)
            endpoint = t0 * noise[..., :7]
            noisy_active = (1.0 - sample_s[:, None, None]) * active + sample_s[:, None, None] * endpoint
            time = t0 * sample_s
            target_velocity = (endpoint - active) / t0
        model_input = jnp.zeros((batch, self.action_horizon, self.action_dim), jnp.float32)
        model_input = model_input.at[..., :7].set(noisy_active)
        if use_primitives:
            model_input = model_input.at[..., 7:14].set(jnp.asarray(dmp_draft, jnp.float32))
        pred_velocity, balance = self._training_forward(
            observation, model_input, time, route_for_flow,
            injection_mode=self.dmp_config.injection_mode if use_primitives else "none",
            train=train,
        )
        flow = jnp.mean(jnp.square(pred_velocity[..., :7] - target_velocity))
        total = flow + route_loss + jnp.mean(balance)
        return total, {
            "flow": flow,
            "route": route_loss,
            "router_ce": ce if use_primitives else jnp.asarray(0.0),
            "router_goal_mse": reg_g if use_primitives else jnp.asarray(0.0),
            "router_tau_mse": reg_tau if use_primitives else jnp.asarray(0.0),
            "router_phase_mse": reg_phase if use_primitives else jnp.asarray(0.0),
            "balance": jnp.mean(balance),
            "bridge_t0": jnp.asarray(t0, jnp.float32),
        }

    def _training_forward(self, observation, model_input, time, route, *, injection_mode, train):
        batch = model_input.shape[0]
        prefix, pm, pa = self.embed_prefix(observation)
        suffix, sm, sa, adarms = super().embed_suffix(observation, model_input, time)
        skill, skill_mask, delta = self._router_condition(route, injection_mode)
        if delta is not None:
            adarms = adarms + delta.astype(adarms.dtype)
        # Encode the frozen VLM prefix separately, as in inference, then detach
        # its KV cache so only action-expert and S3 parameters retain a gradient
        # tape. The route still conditions the action suffix and is supervised
        # directly by the self-decomposition losses above.
        prefix_attn = make_attn_mask(pm, pa)
        prefix_positions = jnp.cumsum(pm, axis=-1) - 1
        (_, _), kv_cache, prefix_balance = self.PaliGemma.llm(
            [prefix, None],
            skill_embeds=skill,
            skill_mask=skill_mask,
            mask=prefix_attn,
            positions=prefix_positions,
            adarms_cond=[None, None],
            deterministic=not train,
        )
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        suffix_attn = make_attn_mask(sm, jnp.broadcast_to(sa, sm.shape))
        prefix_to_suffix = jnp.broadcast_to(pm[:, None, :], (batch, suffix.shape[1], pm.shape[1]))
        full_attn = jnp.concatenate([prefix_to_suffix, suffix_attn], axis=-1)
        suffix_positions = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, out), _, suffix_balance = self.PaliGemma.llm(
            [None, suffix],
            skill_embeds=skill,
            skill_mask=skill_mask,
            mask=full_attn,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms],
            deterministic=not train,
        )
        # Only the action stream is a MoE for pi05_libero_moe_skill_4_90.
        # The prefix-only pass has no active MoE block, and Gemma's empty
        # block-balance reduction is NaN; the action-stream auxiliary is the
        # meaningful load-balancing term for this objective.
        return self.action_out_proj(out[:, -self.action_horizon :]), suffix_balance


def optax_softmax_cross_entropy(logits, labels):
    """Per-example hard-label CE (kept local to avoid an Optax model dependency)."""
    labels = jnp.asarray(labels, jnp.int32)
    log_probs = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
    return -jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]


def dmp_trainable_filter():
    """Train action expert + S3 router/condition projections; PaliGemma stays frozen."""
    return nnx.All(
        nnx.Param,
        nnx_utils_path_regex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj|dmp_router|dmp_shape_encoder|dmp_condition).*"),
    )


def dmp_g0_trainable_filter():
    """Matched G0 update set without primitive router or injection parameters."""
    return nnx.All(
        nnx.Param,
        nnx_utils_path_regex(r".*(llm.*_1|action_in_proj|time_mlp|action_out_proj).*"),
    )


def nnx_utils_path_regex(pattern: str):
    # Lazy import avoids pulling policy/training utilities into pure bridge tests.
    from openpi.shared import nnx_utils

    return nnx_utils.PathRegex(pattern)
