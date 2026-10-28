"""SkillNet Pi0 (MoE) conditioned on a per-chunk continuous skill latent z.

Training:  z ~ q(z | clean action chunk, proprio)       (PosteriorEncoder, reparameterized)
Inference: z = external z (memory blend) if given, else the mean of / a sample from p(z | observation) (ObservationPrior)
Loss:      flow matching (decoder reconstruction) + MoE balance + kl_beta * KL(q || p)

z enters the action expert via (1) the MoE router cross-attention (z tokens in place of, or appended to, skill-ID
embeddings) and/or (2) one extra suffix token placed in the state-token block. With z_dim == 0 every method
delegates to the base `Pi0`, so the existing skill-ID path is untouched. See examples/emergent/DESIGN.md.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config_skill_latent
from openpi.models import pi0_moe_skill
from openpi.models import skill_latent as _sl
import openpi.models.gemma_moe_skill as _gemma
from openpi.shared import array_typing as at

make_attn_mask = pi0_moe_skill.make_attn_mask


class Pi0SkillLatent(pi0_moe_skill.Pi0):
    def __init__(self, config: pi0_config_skill_latent.Pi0SkillLatentConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.z_dim = config.z_dim
        self.z_enabled = config.z_enabled
        self.z_in_router = config.z_in_router
        self.z_in_token = config.z_in_token
        self.use_skill_ids = config.use_skill_ids
        self.kl_beta = config.kl_beta
        self.balance_loss_weight = config.balance_loss_weight
        self.kl_free_bits = config.kl_free_bits
        self.kl_balance = config.kl_balance
        self.prior_pool = config.prior_pool
        self.prior_stop_gradient = config.prior_stop_gradient
        self.z_inference = config.z_inference
        self.embed_dtype = config.dtype
        self.expert_depth = action_expert_config.depth
        self.router_key_dim = config.skill_embed_dim + (4 if config.positional_encoding else 0)
        if not self.z_enabled:
            self.skill_latent = None
            return
        if paligemma_config.moe:
            raise ValueError("Pi0SkillLatent assumes a non-MoE PaliGemma (z only conditions the action expert)")
        if self.z_in_router and not action_expert_config.moe:
            raise ValueError(f"z_inject={config.z_inject!r} needs an MoE action expert")
        state_dim = config.action_dim  # state is padded to action_dim by PadStatesAndActions
        self.skill_latent = _sl.SkillLatent(
            encoder=_sl.PosteriorEncoder(
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
                state_dim=state_dim,
                z_dim=config.z_dim,
                hidden=config.enc_hidden,
                logvar_min=config.logvar_min,
                logvar_max=config.logvar_max,
                rngs=rngs,
            ),
            prior=_sl.ObservationPrior(
                vlm_width=paligemma_config.width,
                state_dim=state_dim,
                use_state=config.prior_use_state,
                z_dim=config.z_dim,
                hidden=config.prior_hidden,
                logvar_min=config.logvar_min,
                logvar_max=config.logvar_max,
                rngs=rngs,
            ),
            router_tokens=(
                _sl.ZRouterTokens(
                    z_dim=config.z_dim, num_tokens=config.z_router_tokens, key_dim=self.router_key_dim, rngs=rngs
                )
                if self.z_in_router
                else None
            ),
            token=(
                _sl.ZToken(
                    z_dim=config.z_dim,
                    hidden=config.z_token_hidden or action_expert_config.width,
                    width=action_expert_config.width,
                    rngs=rngs,
                )
                if self.z_in_token
                else None
            ),
        )

    # ------------------------------------------------------------------ z pieces
    def encode_posterior(self, actions: _model.Actions, state: at.Array):
        """q(z | clean action chunk, proprio) -> (mean, logvar), each [B, z_dim]. Actions must be normalized/padded."""
        return self.skill_latent.encoder(actions, state)

    def _pool_prefix(self, prefix_out, prefix_mask, observation):
        mask = prefix_mask
        if self.prior_pool == "image":
            n_text = 0 if observation.tokenized_prompt is None else observation.tokenized_prompt.shape[1]
            n_image = prefix_out.shape[1] - n_text
            mask = jnp.logical_and(mask, (jnp.arange(prefix_out.shape[1]) < n_image)[None, :])
        pooled = _sl.masked_mean(prefix_out, mask)
        return jax.lax.stop_gradient(pooled) if self.prior_stop_gradient else pooled

    def _prior_from_prefix(self, prefix_out, prefix_mask, observation):
        pooled = self._pool_prefix(prefix_out, prefix_mask, observation)
        mean, logvar = self.skill_latent.prior(pooled, observation.state)
        return mean, logvar, pooled

    def _skill_id_embeds(self, observation):
        """Exactly the base skill-ID embedding path (incl. G1 override and optional positional encoding)."""
        skill_embeds = self.PaliGemma.llm(observation.skills, method="skill_embed")
        skill_embeds = pi0_moe_skill._apply_skill_embed_override(skill_embeds, observation.skill_embed_override)
        if self.positional_encoding:
            b, t = observation.skills.shape
            position = jnp.broadcast_to(jnp.arange(t)[None, :], (b, t))
            position_embeds = self.PaliGemma.llm(position, method="position_embed")
            skill_embeds = jnp.concatenate([skill_embeds, position_embeds], axis=-1)
        return skill_embeds, observation.skill_mask

    def _router_inputs(self, observation, z):
        """Keys/values and mask for the MoE router cross-attention."""
        batch = observation.state.shape[0]
        parts, masks = [], []
        if self.use_skill_ids:
            ids, ids_mask = self._skill_id_embeds(observation)
            parts.append(ids)
            masks.append(ids_mask)
        if self.z_in_router:
            z_tokens = self.skill_latent.router_tokens(z).astype(self.embed_dtype)
            parts.append(z_tokens)
            masks.append(jnp.ones(z_tokens.shape[:2], dtype=jnp.bool_))
        if not parts:
            # "token" injection without skill IDs: a single masked-out key -> exact zero router input
            # (uniform router), matching the behaviour of an all-masked skill sequence.
            parts.append(jnp.zeros((batch, 1, self.router_key_dim), dtype=self.embed_dtype))
            masks.append(jnp.zeros((batch, 1), dtype=jnp.bool_))
        parts = [p.astype(self.embed_dtype) for p in parts]
        return jnp.concatenate(parts, axis=1), jnp.concatenate(masks, axis=1)

    def _embed_suffix_z(self, observation, noisy_actions, timestep, z):
        """Base suffix + (optionally) one z token in the state-token block. Attention-mask rules are unchanged."""
        tokens, input_mask, ar_mask, adarms_cond = self.embed_suffix(observation, noisy_actions, timestep)
        if not self.z_in_token:
            return tokens, input_mask, ar_mask, adarms_cond
        z_token = self.skill_latent.token(z).astype(tokens.dtype)
        if self.pi05:
            # No state token in the pi05 suffix: z forms the conditioning block before the actions
            # (prefix cannot attend to it; it cannot attend to actions; actions attend to it).
            idx, z_ar = 0, True
        else:
            # Join the state token's block (ar=False continues the block opened by the state token).
            idx, z_ar = 1, False
        tokens = jnp.concatenate([tokens[:, :idx], z_token, tokens[:, idx:]], axis=1)
        input_mask = jnp.concatenate(
            [input_mask[:, :idx], jnp.ones((input_mask.shape[0], 1), dtype=jnp.bool_), input_mask[:, idx:]], axis=1
        )
        ar_mask = jnp.concatenate([ar_mask[:idx], jnp.array([z_ar]), ar_mask[idx:]], axis=0)
        return tokens, input_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------ training
    def compute_loss_terms(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        """Returns (recon [*b, ah], terms) where terms has balance_loss, kl (raw), kl_loss, aux_loss (scalars)."""
        if not self.z_enabled:
            recon, balance_loss = super().compute_loss(rng, observation, actions, train=train)
            zero = jnp.zeros((), jnp.float32)
            return recon, {"balance_loss": balance_loss, "kl": zero, "kl_loss": zero, "aux_loss": balance_loss}

        preprocess_rng, noise_rng, time_rng, z_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Posterior from the *clean* chunk; independent of the diffusion noise and time.
        q_mean, q_logvar = self.encode_posterior(actions, observation.state)
        z = _sl.reparameterize(z_rng, q_mean, q_logvar) if train else q_mean

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_suffix_z(observation, x_t, time, z)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        skill_embeds, skill_mask = self._router_inputs(observation, z)

        (prefix_out, suffix_out), _, balance_loss = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            skill_embeds=skill_embeds,
            skill_mask=skill_mask,
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        recon = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Prefix outputs never attend to the suffix, so they equal the inference-time prefix pass.
        p_mean, p_logvar, _ = self._prior_from_prefix(prefix_out, prefix_mask, observation)
        kl_loss, kl_raw = _sl.kl_objective(
            q_mean, q_logvar, p_mean, p_logvar, free_bits=self.kl_free_bits, balance=self.kl_balance
        )
        aux_loss = self.balance_loss_weight * balance_loss + self.kl_beta * kl_loss
        return recon, {"balance_loss": balance_loss, "kl": kl_raw, "kl_loss": kl_loss, "aux_loss": aux_loss}

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        """Same contract as the base model: (per-step flow-matching loss, scalar aux loss incl. beta * KL)."""
        if not self.z_enabled:
            return super().compute_loss(rng, observation, actions, train=train)
        recon, terms = self.compute_loss_terms(rng, observation, actions, train=train)
        return recon, terms["aux_loss"]

    # ------------------------------------------------------------------ inference
    def _prefix_pass(self, observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # PaliGemma is not MoE, so the prefix pass needs no router input.
        (prefix_out, _), kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None], skill_embeds=None, skill_mask=None, mask=prefix_attn_mask, positions=positions
        )
        return prefix_tokens, prefix_mask, prefix_out, kv_cache

    def memory_key_features(self, observation: _model.Observation, *, neutral_prompt=None):
        """Features for memory keys (not used by training). Returns a dict of float32 arrays:

        - "image_pool_no_language" [B, W]: masked mean of image-token outputs from a *separate* VLM pass without the
          task language (``neutral_prompt=None`` drops the language tokens; or pass ``(tokens [B, L], mask [B, L])``
          of a fixed generic sentence). pi0.5's prefix attends bidirectionally, so this second pass is required for a
          language-free key. Main memory key.
        - "image_pool" [B, W]: masked mean of image-token outputs from the normal input (image + task language).
        - "state" [B, S]: proprio as fed to the model.
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        def image_pool(obs):
            _, prefix_mask, prefix_out, _ = self._prefix_pass(obs)
            n_text = 0 if obs.tokenized_prompt is None else obs.tokenized_prompt.shape[1]
            n_image = prefix_out.shape[1] - n_text
            mask = jnp.logical_and(prefix_mask, (jnp.arange(prefix_out.shape[1]) < n_image)[None, :])
            return _sl.masked_mean(prefix_out, mask)

        if neutral_prompt is None:
            no_lang = dataclasses.replace(observation, tokenized_prompt=None, tokenized_prompt_mask=None)
        else:
            tokens, mask = neutral_prompt
            no_lang = dataclasses.replace(
                observation, tokenized_prompt=jnp.asarray(tokens), tokenized_prompt_mask=jnp.asarray(mask, bool)
            )
        return {
            "image_pool_no_language": image_pool(no_lang),
            "image_pool": image_pool(observation),
            "state": observation.state.astype(jnp.float32),
        }

    def infer_prior(self, observation: _model.Observation):
        """p(z | observation) -> (mean, logvar, pooled VLM feature). For memory keys / analysis."""
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_mask, prefix_out, _ = self._prefix_pass(observation)
        return self._prior_from_prefix(prefix_out, prefix_mask, observation)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        capture_auxiliary: bool = False,
        z: at.Float[at.Array, "b zd"] | None = None,
    ) -> _model.Actions:
        """z priority: `z` argument > `observation.skill_z` > prior (mean or sample, per `z_inference`)."""
        if not self.z_enabled:
            if z is not None:
                raise ValueError("z given but the model has z_dim=0")
            return super().sample_actions(
                rng, observation, num_steps=num_steps, noise=noise, capture_auxiliary=capture_auxiliary
            )
        observation = _model.preprocess_observation(None, observation, train=False)
        noise_rng, z_rng = jax.random.split(rng)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_out, kv_cache = self._prefix_pass(observation)
        if z is None:
            z = observation.skill_z
        if z is None:
            p_mean, p_logvar, _ = self._prior_from_prefix(prefix_out, prefix_mask, observation)
            z = _sl.reparameterize(z_rng, p_mean, p_logvar) if self.z_inference == "prior_sample" else p_mean
        z = jnp.asarray(z, jnp.float32)
        skill_embeds, skill_mask = self._router_inputs(observation, z)

        def step(carry):
            x_t, time = carry[:2]
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_suffix_z(
                observation, x_t, jnp.broadcast_to(time, batch_size), z
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            llm_output = self.PaliGemma.llm(
                [None, suffix_tokens],
                skill_embeds=skill_embeds,
                skill_mask=skill_mask,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                capture_auxiliary=capture_auxiliary,
            )
            if capture_auxiliary:
                (prefix_out_, suffix_out), _, _, layer_aux = llm_output
            else:
                (prefix_out_, suffix_out), _, _ = llm_output
            assert prefix_out_ is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if capture_auxiliary:
                layer_features, router_top1, selected_norm, shared_norm = layer_aux
                return x_t + dt * v_t, time + dt, layer_features, router_top1, selected_norm, shared_norm
            return x_t + dt * v_t, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        if capture_auxiliary:
            suffix_tokens, _, _, _ = self._embed_suffix_z(observation, noise, jnp.ones(batch_size), z)
            init_features = jnp.zeros((self.expert_depth,) + suffix_tokens.shape, dtype=self.embed_dtype)
            init_routes = jnp.zeros((self.expert_depth, batch_size, suffix_tokens.shape[1]), dtype=jnp.int32)
            init_norms = jnp.zeros((self.expert_depth, batch_size, suffix_tokens.shape[1]), dtype=jnp.float32)
            x_0, _, feats, routes, sel, shared = jax.lax.while_loop(
                cond, step, (noise, 1.0, init_features, init_routes, init_norms, init_norms)
            )
            return x_0, feats, routes, sel, shared
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
