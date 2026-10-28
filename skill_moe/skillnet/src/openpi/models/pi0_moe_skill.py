import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config_moe_skill
import openpi.models.gemma_moe_skill as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _apply_skill_embed_override(skill_embeds, override):
    """Return original embeddings verbatim unless the pilot hook is explicitly set."""
    return skill_embeds if override is None else override


def _apply_memory_condition(action_tokens, adarms_cond, observation, adapter):
    """Inject the selected memory residual into the current action suffix."""
    if not observation.memory_enabled:
        return action_tokens, adarms_cond
    if adapter is None:
        raise ValueError("memory_enabled=True requires an attached memory injection adapter")

    mode = adapter.mode
    if mode == "action_tokens":
        if observation.memory_features is None:
            raise ValueError("action_tokens injection requires memory_features [B, Dz]")
        residual = adapter(c=observation.memory_features, horizon=action_tokens.shape[1])
        return action_tokens + jnp.asarray(residual, dtype=action_tokens.dtype), adarms_cond
    if mode == "adarms_cond":
        if observation.memory_features is None:
            raise ValueError("adarms_cond injection requires memory_features [B, Dz]")
        if adarms_cond is None:
            raise ValueError("adarms_cond injection requires a Pi05 model with adaRMS")
        residual = adapter(c=observation.memory_features)
        return action_tokens, adarms_cond + jnp.asarray(residual, dtype=adarms_cond.dtype)
    if mode == "candidate_attention":
        if observation.memory_candidates is None or observation.memory_candidate_mask is None:
            raise ValueError("candidate_attention requires memory_candidates and memory_candidate_mask")
        residual = adapter(
            action_tokens=action_tokens,
            candidates=observation.memory_candidates,
            candidate_mask=observation.memory_candidate_mask,
        )
        return action_tokens + jnp.asarray(residual, dtype=action_tokens.dtype), adarms_cond
    raise ValueError(f"Unsupported memory injection adapter mode: {mode!r}")


def subtask_generated_lengths(tokens: at.Int[at.Array, "b n"], eos_token: int = 1):
    """Length through the first EOS (inclusive), or full width if EOS is absent."""
    eos = tokens == eos_token
    first_eos = jnp.argmax(eos, axis=-1)
    return jnp.where(jnp.any(eos, axis=-1), first_eos + 1, tokens.shape[-1])


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config_moe_skill.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.positional_encoding=config.positional_encoding
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                skill_num=config.skill_num,
                skill_embed_dim=config.skill_embed_dim,
                positional_encoding=config.positional_encoding
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=config.vision_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # Kept absent from released checkpoint parameters. Attach a pilot adapter
        # after restoring the frozen base weights and before building Policy/JIT.
        self.memory_injection = None

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def attach_memory_injection(self, adapter: nnx.Module) -> None:
        """Attach a trainable pilot adapter after restoring frozen base weights."""
        self.memory_injection = adapter

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        image_token_count = 0
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            image_token_count += image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Normal action prompts are bidirectional. Subtask targets carry a
            # per-token prefix/causal mask from the tokenizer.
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        if obs.token_ar_mask is None:
            ar_mask = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=jnp.bool_)
        else:
            image_ar_mask = jnp.zeros((tokens.shape[0], image_token_count), dtype=jnp.bool_)
            ar_mask = jnp.concatenate([image_ar_mask, obs.token_ar_mask.astype(jnp.bool_)], axis=1)
        return tokens, input_mask, ar_mask

    def build_full_observation(self, observation: _model.Observation, subtask_tokens: at.Int[at.Array, "b n"]):
        """Append generated subtask tokens after the Task/Subtask prefix."""
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Subtask decoding requires tokenized_prompt and its mask")
        token_len = observation.tokenized_prompt.shape[-1]
        prefix_len = jnp.sum(observation.tokenized_prompt_mask, axis=-1).astype(jnp.int32)
        token_positions = prefix_len[:, None] + jnp.arange(subtask_tokens.shape[-1])[None, :]
        gen_len = subtask_generated_lengths(subtask_tokens)
        generated_valid = jnp.arange(subtask_tokens.shape[-1])[None, :] < gen_len[:, None]
        in_bounds = token_positions < token_len
        one_hot = jax.nn.one_hot(jnp.clip(token_positions, 0, token_len - 1), token_len)
        one_hot = one_hot * (generated_valid & in_bounds)[..., None]
        appended = jnp.sum(one_hot * subtask_tokens[..., None], axis=1).astype(observation.tokenized_prompt.dtype)
        prefix_only = jnp.where(
            jnp.arange(token_len)[None, :] < prefix_len[:, None], observation.tokenized_prompt, 0
        )
        full_tokens = prefix_only + appended
        full_mask = (jnp.arange(token_len)[None, :] < prefix_len[:, None]) | jnp.any(one_hot > 0, axis=1)
        return observation.replace(
            tokenized_prompt=full_tokens,
            tokenized_prompt_mask=full_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def generate_subtask(
        self,
        observation: _model.Observation,
        *,
        max_tokens: int = 50,
    ) -> at.Int[at.Array, "b n"]:
        """Greedy PaliGemma decoding with a static preallocated KV cache.

        EOS (token id 1) is included in the returned sequence and stops decoding;
        the remaining output slots are zero padded. `max_tokens` includes EOS.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        batch_size, prefix_size, _ = prefix_tokens.shape
        max_tokens = min(int(max_tokens), observation.tokenized_prompt.shape[-1])
        prefix_lengths = jnp.sum(prefix_mask, axis=-1).astype(jnp.int32)
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_tokens)))
        (prefix_out, _), kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            skill_embeds=None,
            skill_mask=None,
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        last_positions = jnp.maximum(prefix_lengths - 1, 0)
        last_hidden = jnp.take_along_axis(prefix_out, last_positions[:, None, None], axis=1)
        last_logits = self.PaliGemma.llm(pre_logits=last_hidden, method="logits")
        total_cache_size = prefix_size + max_tokens
        generated = jnp.zeros((batch_size, max_tokens), dtype=jnp.int32)
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        key_positions = jnp.arange(total_cache_size)[None, :]

        def condition(carry):
            step, _, _, _, done = carry
            return (step < max_tokens) & ~jnp.all(done)

        def body(carry):
            step, logits, cache, output, done = carry
            token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
            token = jnp.where(done, 0, token)
            output = output.at[:, step].set(token)
            done = done | (token == 1)
            token_embed = self.PaliGemma.llm(token[:, None], method="embed")
            token_position = prefix_lengths + step
            valid_keys = key_positions < (token_position + 1)[:, None]
            decode_mask = valid_keys[:, None, :]
            (_, decoded), cache, _ = self.PaliGemma.llm(
                [token_embed, None],
                skill_embeds=None,
                skill_mask=None,
                mask=decode_mask,
                positions=token_position[:, None],
                kv_cache=cache,
            )
            next_logits = self.PaliGemma.llm(pre_logits=decoded, method="logits")
            return step + 1, next_logits, cache, output, done

        _, _, _, generated, _ = jax.lax.while_loop(
            condition,
            body,
            (jnp.asarray(0, jnp.int32), last_logits, kv_cache, generated, finished),
        )
        return generated

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        action_expert_tokens, adarms_cond = _apply_memory_condition(
            action_expert_tokens, adarms_cond, obs, self.memory_injection
        )

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if observation.token_loss_mask is not None:
            # Subtask-only training is a PaliGemma next-token objective. The
            # action stream is absent from this forward pass, so action experts
            # receive no task gradients.
            del actions
            preprocess_rng = rng if train else None
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            input_tokens = prefix_tokens[:, :-1]
            input_mask = prefix_mask[:, :-1]
            attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)[:, :-1, :-1]
            positions = jnp.cumsum(prefix_mask, axis=1)[:, :-1] - 1
            (prefix_out, _), _, _ = self.PaliGemma.llm(
                [input_tokens, None],
                skill_embeds=None,
                skill_mask=None,
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, None],
                deterministic=not train,
            )
            # Image tokens are part of the prefix context but have no language
            # targets. Align each next-token target with the preceding language
            # state, starting after the visual prefix.
            text_start = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
            text_pre_logits = prefix_out[:, text_start:]
            logits = self.PaliGemma.llm(pre_logits=text_pre_logits, method="logits")
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            targets = observation.tokenized_prompt[:, 1:]
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)
            token_counts = jnp.sum(loss_mask, axis=-1)
            loss = -jnp.sum(target_log_probs * loss_mask, axis=-1) / jnp.maximum(token_counts, 1.0)
            return loss, jnp.zeros((), dtype=jnp.float32)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # get skill_embeddings
        skill_embeds = self.PaliGemma.llm(observation.skills, method="skill_embed")
        skill_embeds = _apply_skill_embed_override(skill_embeds, observation.skill_embed_override)
        # get positianl embedding
        if self.positional_encoding:
            B, T = observation.skills.shape
            position_ = jnp.arange(T)[None, :]  # shape (1,1,m)
            position_ = jnp.broadcast_to(position_, (B, T))
            position_embeds = self.PaliGemma.llm(position_, method="position_embed")
            # cat skill and object embeddings
            skill_embeds = jnp.concatenate([skill_embeds, position_embeds], axis=-1)

        (prefix_out, suffix_out), _, balance_loss = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], skill_embeds=skill_embeds, skill_mask=observation.skill_mask, mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # (prefix_out, suffix_out), _, balance_loss = self.PaliGemma.llm(
        #     [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        # )
        # jax.debug.print("balance_loss: {}", balance_loss)
        # main loss
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        main_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        return main_loss, balance_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        capture_auxiliary: bool = False,
        capture_denoising_trace: bool = False,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        skill_embeds = self.PaliGemma.llm(observation.skills, method="skill_embed")
        skill_embeds = _apply_skill_embed_override(skill_embeds, observation.skill_embed_override)

        if self.positional_encoding:
            B, T = observation.objects.shape
            position_ = jnp.arange(T)[None, :]  # shape (1,1,m)
            position_ = jnp.broadcast_to(position_, (B, T))
            position_embeds = self.PaliGemma.llm(position_, method="position_embed")
            # cat skill and object embeddings
            skill_embeds = jnp.concatenate([skill_embeds, position_embeds], axis=-1)
            
        _, kv_cache, balance_loss = self.PaliGemma.llm([prefix_tokens, None], skill_embeds=skill_embeds, skill_mask=observation.skill_mask, mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry[:2]
            trace_index = carry[2] if capture_denoising_trace else None
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            llm_output = self.PaliGemma.llm(
                [None, suffix_tokens], 
                skill_embeds=skill_embeds, 
                skill_mask=observation.skill_mask,
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                capture_auxiliary=(capture_auxiliary or capture_denoising_trace),
            )
            if capture_auxiliary or capture_denoising_trace:
                (prefix_out, suffix_out), _, balance_loss, layer_aux = llm_output
            else:
                (prefix_out, suffix_out), _, balance_loss = llm_output
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if capture_denoising_trace:
                layer_features, _, _, _ = layer_aux
                selected_layers = jnp.asarray([9, 17], dtype=jnp.int32)
                # Record only pooled action-suffix activations to keep the
                # denoising trace compact: [selected layer, batch, width].
                pooled = jnp.mean(layer_features[selected_layers, :, -self.action_horizon :, :].astype(jnp.float32), axis=2)
                x0_hat = x_t - time * v_t
                return (
                    x_t + dt * v_t,
                    time + dt,
                    trace_index + 1,
                    carry[3].at[trace_index].set(x_t),
                    carry[4].at[trace_index].set(x0_hat),
                    carry[5].at[trace_index].set(pooled),
                )

            if capture_auxiliary:
                layer_features, router_top1, selected_norm, shared_norm = layer_aux
                return x_t + dt * v_t, time + dt, layer_features, router_top1, selected_norm, shared_norm
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry[:2]
            # robust to floating-point error
            return time >= -dt / 2

        if capture_denoising_trace:
            if num_steps < 1:
                raise ValueError("num_steps must be positive")
            suffix_tokens, _, _, _ = self.embed_suffix(observation, noise, jnp.ones(batch_size))
            depth = _gemma.get_config("gemma_300m_moe_4").depth
            if depth <= 17:
                raise ValueError(f"denoising trace requires layer 17, model depth is {depth}")
            action_width = suffix_tokens.shape[-1]
            trace_shape = (num_steps, batch_size, self.action_horizon, self.action_dim)
            hidden_shape = (num_steps, 2, batch_size, action_width)
            x_0, _, _, x_trace, x0_trace, hidden_trace = jax.lax.while_loop(
                cond,
                step,
                (
                    noise,
                    1.0,
                    jnp.asarray(0, jnp.int32),
                    jnp.zeros(trace_shape, dtype=noise.dtype),
                    jnp.zeros(trace_shape, dtype=noise.dtype),
                    jnp.zeros(hidden_shape, dtype=jnp.float32),
                ),
            )
            times = 1.0 - jnp.arange(num_steps, dtype=jnp.float32) / num_steps
            return x_0, x_trace, x0_trace, hidden_trace, times

        if capture_auxiliary:
            suffix_tokens, _, _, _ = self.embed_suffix(observation, noise, jnp.ones(batch_size))
            # Shapes match the auxiliary values returned by each denoising layer.
            # The pilot targets the fixed gemma_300m_moe_4 checkpoint (18 layers).
            depth = _gemma.get_config("gemma_300m_moe_4").depth
            init_features = jnp.zeros((depth,) + suffix_tokens.shape, dtype=jnp.bfloat16)
            init_routes = jnp.zeros((depth, suffix_tokens.shape[0], suffix_tokens.shape[1]), dtype=jnp.int32)
            init_norms = jnp.zeros((depth, suffix_tokens.shape[0], suffix_tokens.shape[1]), dtype=jnp.float32)
            x_0, _, last_features, last_routes, last_selected_norm, last_shared_norm = jax.lax.while_loop(
                cond, step, (noise, 1.0, init_features, init_routes, init_norms, init_norms)
            )
            return x_0, last_features, last_routes, last_selected_norm, last_shared_norm
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
