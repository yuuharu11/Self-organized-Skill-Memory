"""Independent Stage-A Optimus-R-style path for SkillNet π0.5.

This is an openpi reimplementation, not the authors' code.  It adds memory
seed tokens, separate query/skill readouts, action reconstruction, query NCE,
and a skill-token prefix condition without changing existing Pi0 classes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config_moe_skill as _base_config
from openpi.models.pi0_moe_skill import Pi0 as _SkillNetPi0
from openpi.models.pi0_moe_skill import make_attn_mask, posemb_sincos
from openpi.models.pi0_moe_skill import _apply_skill_embed_override
from openpi.shared import array_typing as at


def query_contrastive_loss(query, task_ids, progress, episode_ids, sample_indices, *, temperature: float):
    """Multi-positive InfoNCE with the paper's task/progress and trajectory rules."""
    query = query.astype(jnp.float32)
    query = query / jnp.maximum(jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-8)
    similarity = (query @ query.T) / temperature
    size = query.shape[0]
    nonself = ~jnp.eye(size, dtype=jnp.bool_)
    same_task_near_progress = (task_ids[:, None] == task_ids[None, :]) & (
        jnp.abs(progress[:, None] - progress[None, :]) <= 0.1
    )
    same_trajectory_near_time = (episode_ids[:, None] == episode_ids[None, :]) & (
        jnp.abs(sample_indices[:, None] - sample_indices[None, :]) <= 6
    )
    positive = (same_task_near_progress | same_trajectory_near_time) & nonself
    valid_anchor = jnp.any(positive, axis=-1)
    denominator_logits = jnp.where(nonself, similarity, -1e9)
    positive_logits = jnp.where(positive, similarity, -1e9)
    log_denominator = jax.nn.logsumexp(denominator_logits, axis=-1)
    log_positive = jax.nn.logsumexp(positive_logits, axis=-1)
    per_anchor = jnp.where(valid_anchor, log_denominator - log_positive, 0.0)
    return jnp.sum(per_anchor) / jnp.maximum(jnp.sum(valid_anchor), 1)


def retrieve_topk(query, bank_query, bank_skill, *, top_k: int = 2, temperature: float = 0.1):
    """Cosine top-k retrieval followed by a softmax weighted skill prototype."""
    if top_k < 1 or temperature <= 0:
        raise ValueError("top_k and retrieval temperature must be positive")
    query = jnp.asarray(query, jnp.float32)
    bank_query = jnp.asarray(bank_query, jnp.float32)
    bank_skill = jnp.asarray(bank_skill, jnp.float32)
    if bank_query.ndim != 2 or bank_skill.ndim != 2 or bank_query.shape[0] != bank_skill.shape[0]:
        raise ValueError("bank query and skill arrays must be aligned rank-2 arrays")
    if top_k > bank_query.shape[0]:
        raise ValueError(f"top_k={top_k} exceeds bank size {bank_query.shape[0]}")
    query = query / jnp.maximum(jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-8)
    bank_query = bank_query / jnp.maximum(jnp.linalg.norm(bank_query, axis=-1, keepdims=True), 1e-8)
    scores = query @ bank_query.T
    top_scores, indices = jax.lax.top_k(scores, top_k)
    weights = jax.nn.softmax(top_scores / temperature, axis=-1)
    candidates = bank_skill[indices]
    weighted_skill = jnp.sum(candidates * weights[..., None], axis=1)
    return weighted_skill, indices, weights, top_scores


class Pi0OptimusR(_SkillNetPi0):
    """SkillNet π0.5 with a trainable memory-token and skill-token path."""

    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        from openpi.models import gemma_moe_skill as _gemma

        self.optimus_config = config
        self.memory_token_count = int(config.memory_token_count)
        self.skill_token_count = int(config.skill_token_count)
        width = _gemma.get_config(config.paligemma_variant).width
        scale = width**-0.5
        self.memory_seed_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (self.memory_token_count, width), dtype=jnp.float32) * 0.02
        )
        self.optimus_state_token = nnx.Linear(config.action_dim, width, rngs=rngs)
        self.query_pool_query = nnx.Param(jax.random.normal(rngs.params(), (width,), dtype=jnp.float32) * scale)
        self.query_pool_matrix = nnx.Param(jax.random.normal(rngs.params(), (width, width), dtype=jnp.float32) * scale)
        self.skill_pool_query = nnx.Param(jax.random.normal(rngs.params(), (width,), dtype=jnp.float32) * scale)
        self.skill_pool_matrix = nnx.Param(jax.random.normal(rngs.params(), (width, width), dtype=jnp.float32) * scale)

        self.query_encoder_in = nnx.Linear(width, config.interface_hidden_dim, rngs=rngs)
        self.query_encoder_out = nnx.Linear(config.interface_hidden_dim, config.query_dim, rngs=rngs)
        self.skill_encoder_in = nnx.Linear(width, config.interface_hidden_dim, rngs=rngs)
        self.skill_encoder_out = nnx.Linear(config.interface_hidden_dim, config.skill_dim, rngs=rngs)
        self.skill_decoder_in = nnx.Linear(config.skill_dim, config.interface_hidden_dim, rngs=rngs)
        self.skill_decoder_out = nnx.Linear(
            config.interface_hidden_dim, config.action_horizon * config.action_dim, rngs=rngs
        )
        self.skill_token_projector = nnx.Linear(
            config.skill_dim, config.skill_token_count * width, rngs=rngs
        )

    def _router_embedding(self, observation: _model.Observation):
        skill_embeds = None
        if observation.skills is not None:
            skill_embeds = self.PaliGemma.llm(observation.skills, method="skill_embed")
            skill_embeds = _apply_skill_embed_override(skill_embeds, observation.skill_embed_override)
            if self.positional_encoding:
                if observation.objects is None:
                    raise ValueError("positional encoding requires observation.objects")
                batch, count = observation.objects.shape
                positions = jnp.broadcast_to(jnp.arange(count)[None, :], (batch, count))
                position_embeds = self.PaliGemma.llm(positions, method="position_embed")
                skill_embeds = jnp.concatenate([skill_embeds, position_embeds], axis=-1)
        return skill_embeds

    def _prefix_with_memory(self, observation: _model.Observation):
        # LIBERO-90 S2 exports this third camera as an all-false mask. Omitting
        # its image tokens is exactly equivalent under the attention mask and
        # avoids a redundant SigLIP forward/backward on the shared GPU.
        tokens, masks = [], []
        image_token_count = 0
        for name, image in observation.images.items():
            if name == "right_wrist_0_rgb":
                continue
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            tokens.append(image_tokens)
            masks.append(einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            image_token_count += image_tokens.shape[1]
        if observation.tokenized_prompt is not None:
            tokens.append(self.PaliGemma.llm(observation.tokenized_prompt, method="embed"))
            masks.append(observation.tokenized_prompt_mask)
        prefix = jnp.concatenate(tokens, axis=1)
        mask = jnp.concatenate(masks, axis=1)
        if observation.token_ar_mask is None:
            ar = jnp.zeros(mask.shape, dtype=jnp.bool_)
        else:
            image_ar = jnp.zeros((mask.shape[0], image_token_count), dtype=jnp.bool_)
            ar = jnp.concatenate([image_ar, observation.token_ar_mask.astype(jnp.bool_)], axis=1)
        batch = prefix.shape[0]
        memory_tokens = jnp.broadcast_to(
            self.memory_seed_tokens.value.astype(prefix.dtype)[None],
            (batch, self.memory_token_count, prefix.shape[-1]),
        )
        state_token = self.optimus_state_token(observation.state.astype(jnp.float32))[:, None, :].astype(prefix.dtype)
        pieces = [prefix, state_token, memory_tokens]
        valid = [mask, jnp.ones((batch, 1), dtype=jnp.bool_),
                 jnp.ones((batch, self.memory_token_count), dtype=jnp.bool_)]
        causal = [ar, jnp.zeros((batch, 1 + self.memory_token_count), dtype=jnp.bool_)]
        if observation.memory_features is not None:
            skill = jnp.asarray(observation.memory_features, jnp.float32)
            if skill.shape != (batch, self.optimus_config.skill_dim):
                raise ValueError(
                    f"memory_features must have [batch,{self.optimus_config.skill_dim}], got {skill.shape}"
                )
            projected = self.skill_token_projector(skill)
            skill_tokens = projected.reshape(batch, self.skill_token_count, prefix.shape[-1]).astype(prefix.dtype)
            pieces.append(skill_tokens)
            valid.append(jnp.ones((batch, self.skill_token_count), dtype=jnp.bool_))
            causal.append(jnp.zeros((batch, self.skill_token_count), dtype=jnp.bool_))
        return jnp.concatenate(pieces, axis=1), jnp.concatenate(valid, axis=1), jnp.concatenate(causal, axis=1)

    @override
    def embed_prefix(self, obs: _model.Observation):
        return self._prefix_with_memory(obs)

    def encode_memory(self, observation: _model.Observation, *, train: bool = False):
        """Encode current context into separate query and action-grounded skill vectors."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix, mask, ar = self._prefix_with_memory(observation.replace(memory_features=None))
        attention = make_attn_mask(mask, ar)
        positions = jnp.cumsum(mask, axis=1) - 1
        (outputs, _), _, _ = self.PaliGemma.llm(
            [prefix, None], skill_embeds=self._router_embedding(observation),
            skill_mask=observation.skill_mask, mask=attention, positions=positions,
            adarms_cond=[None, None], deterministic=not train,
        )
        memory_hidden = outputs[:, -self.memory_token_count:, :].astype(jnp.float32)
        query_pooled = self._attention_pool(memory_hidden, self.query_pool_query.value, self.query_pool_matrix.value)
        skill_pooled = self._attention_pool(memory_hidden, self.skill_pool_query.value, self.skill_pool_matrix.value)
        query = self.query_encoder_out(nnx.gelu(self.query_encoder_in(query_pooled)))
        skill = self.skill_encoder_out(nnx.gelu(self.skill_encoder_in(skill_pooled)))
        query = query / jnp.maximum(jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-8)
        return query, skill

    @staticmethod
    def _attention_pool(memory_hidden, pool_query, pool_matrix):
        transformed = jnp.einsum("bmd,df->bmf", memory_hidden, pool_matrix.astype(jnp.float32))
        logits = jnp.einsum("d,bmd->bm", pool_query.astype(jnp.float32), transformed)
        logits = logits / jnp.sqrt(jnp.asarray(memory_hidden.shape[-1], jnp.float32))
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bm,bmd->bd", weights, memory_hidden)

    def reconstruct_actions(self, skill):
        hidden = nnx.gelu(self.skill_decoder_in(skill))
        prediction = self.skill_decoder_out(hidden)
        return prediction.reshape(skill.shape[0], self.action_horizon, self.action_dim)

    def compute_stage_a_loss(
        self, rng, observation, actions, *, task_ids, progress, episode_ids, sample_indices, train: bool = True
    ):
        """Compute L_fm + 0.1 L_query-NCE + 1.0 L_skill-Huber."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        query, skill = self.encode_memory(observation, train=train)
        conditioned = observation.replace(memory_features=skill)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1.0, batch_shape) * 0.999 + 0.001
        expanded_time = time[..., None, None]
        noisy_actions = expanded_time * noise + (1.0 - expanded_time) * actions
        flow_target = noise - actions
        prefix, prefix_mask, prefix_ar = self._prefix_with_memory(conditioned)
        suffix, suffix_mask, suffix_ar, adarms_cond = super().embed_suffix(conditioned, noisy_actions, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        suffix_ar = jnp.broadcast_to(suffix_ar, (actions.shape[0], suffix_ar.shape[-1]))
        causal_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=1)
        attention = make_attn_mask(input_mask, causal_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_output, suffix_output), _, router_loss = self.PaliGemma.llm(
            [prefix, suffix], skill_embeds=self._router_embedding(conditioned),
            skill_mask=conditioned.skill_mask, mask=attention, positions=positions,
            adarms_cond=[None, adarms_cond], deterministic=not train,
        )
        del prefix_output
        velocity = self.action_out_proj(suffix_output[:, -self.action_horizon:])
        flow_loss = jnp.mean(jnp.square(velocity.astype(jnp.float32) - flow_target.astype(jnp.float32))) + router_loss

        reconstructed = self.reconstruct_actions(skill).astype(jnp.float32)
        error = reconstructed - actions.astype(jnp.float32)
        abs_error = jnp.abs(error)
        huber = jnp.where(abs_error <= 1.0, 0.5 * error * error, abs_error - 0.5)
        skill_loss = jnp.mean(huber)
        query_loss = query_contrastive_loss(
            query, task_ids, progress, episode_ids, sample_indices,
            temperature=self.optimus_config.query_contrastive_temperature,
        )
        total = (
            self.optimus_config.fm_loss_weight * flow_loss
            + self.optimus_config.query_loss_weight * query_loss
            + self.optimus_config.skill_reconstruction_weight * skill_loss
        )
        return total, {"flow": flow_loss, "query_nce": query_loss, "skill_huber": skill_loss,
                       "router": router_loss, "valid_query_anchors": jnp.sum(
                           jnp.any((task_ids[:, None] == task_ids[None, :]) &
                                   (jnp.abs(progress[:, None] - progress[None, :]) <= 0.1) &
                                   ~jnp.eye(task_ids.shape[0], dtype=jnp.bool_), axis=-1)
                       )}

    def retrieve_skill_condition(self, observation, bank_query, bank_skill, *, top_k: int = 2,
                                 temperature: float = 0.1):
        query, _ = self.encode_memory(observation, train=False)
        skill, indices, weights, scores = retrieve_topk(
            query, bank_query, bank_skill, top_k=top_k, temperature=temperature
        )
        return observation.replace(memory_features=skill), indices, weights, scores

    def sample_actions_with_bank(self, rng, observation, bank_query, bank_skill, *, top_k: int = 2,
                                 temperature: float = 0.1, **kwargs):
        conditioned, indices, weights, scores = self.retrieve_skill_condition(
            observation, bank_query, bank_skill, top_k=top_k, temperature=temperature
        )
        actions = super().sample_actions(rng, conditioned, **kwargs)
        return actions, {"indices": indices, "weights": weights, "scores": scores}
