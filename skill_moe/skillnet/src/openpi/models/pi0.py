# import dataclasses
# import logging

# import einops
# import flax.nnx as nnx
# import flax.nnx.bridge as nnx_bridge
# import jax
# import jax.numpy as jnp
# from typing_extensions import override

# from openpi.models import model as _model
# import openpi.models.gemma as _gemma
# import openpi.models.siglip as _siglip
# from openpi.shared import array_typing as at
# import openpi.shared.nnx_utils as nnx_utils

# logger = logging.getLogger("openpi")


# def make_attn_mask(input_mask, mask_ar):
#     """Adapted from big_vision.

#     Tokens can attend to valid inputs tokens which have a cumulative mask_ar
#     smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
#     setup several types of attention, for example:

#       [[1 1 1 1 1 1]]: pure causal attention.

#       [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
#           themselves and the last 3 tokens have a causal attention. The first
#           entry could also be a 1 without changing behaviour.

#       [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
#           block can attend all previous blocks and all tokens on the same block.

#     Args:
#       input_mask: bool[B, N] true if its part of the input, false if padding.
#       mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
#         it and false where it shares the same attention mask as the previous token.
#     """
#     mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
#     cumsum = jnp.cumsum(mask_ar, axis=1)
#     attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
#     valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
#     return jnp.logical_and(attn_mask, valid_mask)


# @at.typecheck
# def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
#                   max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
#     """Computes sine-cosine positional embedding vectors for scalar positions."""
#     if embedding_dim % 2 != 0:
#         raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

#     fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
#     period = min_period * (max_period / min_period)**fraction
#     sinusoid_input = jnp.einsum(
#         "i,j->ij",
#         pos,
#         1.0 / period * 2 * jnp.pi,
#         precision=jax.lax.Precision.HIGHEST,
#     )
#     return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# @dataclasses.dataclass(frozen=True)
# class Pi0Config(_model.BaseModelConfig):
#     dtype: str = "bfloat16"
#     paligemma_variant: _gemma.Variant = "gemma_2b"
#     action_expert_variant: _gemma.Variant = "gemma_300m"

#     # Set the model specific defaults.
#     action_dim: int = 32
#     action_horizon: int = 50
#     max_token_len: int = 48

#     @property
#     @override
#     def model_type(self) -> _model.ModelType:
#         return _model.ModelType.PI0

#     @override
#     def create(self, rng: at.KeyArrayLike) -> "Pi0":
#         return Pi0(self, rngs=nnx.Rngs(rng))

#     @override
#     def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
#         image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
#         image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

#         with at.disable_typechecking():
#             observation_spec = _model.Observation(
#                 images={
#                     "base_0_rgb": image_spec,
#                     "left_wrist_0_rgb": image_spec,
#                     "right_wrist_0_rgb": image_spec,
#                 },
#                 image_masks={
#                     "base_0_rgb": image_mask_spec,
#                     "left_wrist_0_rgb": image_mask_spec,
#                     "right_wrist_0_rgb": image_mask_spec,
#                 },
#                 state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
#                 tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
#                 tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
#             )
#         action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

#         return observation_spec, action_spec

#     def get_freeze_filter(self) -> nnx.filterlib.Filter:
#         """Returns the freeze filter based on the model config."""
#         filters = []
#         has_lora = False
#         gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
#         action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
#         if "lora" in self.paligemma_variant:
#             filters.append(gemma_params_filter, )
#             if "lora" not in self.action_expert_variant:
#                 # If only freeze gemma params, exclude action expert params.
#                 filters.append(nnx.Not(action_expert_params_filter), )
#             has_lora = True
#         elif "lora" in self.action_expert_variant:
#             filters.append(action_expert_params_filter, )
#             has_lora = True

#         if has_lora:
#             # If any lora is used, exclude all lora params.
#             filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
#         if not filters:
#             return nnx.Nothing
#         return nnx.All(*filters)


# class Pi0(_model.BaseModel):

#     def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
#         super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
#         paligemma_config = _gemma.get_config(config.paligemma_variant)
#         action_expert_config = _gemma.get_config(config.action_expert_variant)
#         # TODO: rewrite gemma in NNX. For now, use bridge.
#         llm = nnx_bridge.ToNNX(
#             _gemma.Module(
#                 configs=[paligemma_config, action_expert_config],
#                 embed_dtype=config.dtype,
#             ))
#         llm.lazy_init(rngs=rngs, method="init")
#         img = nnx_bridge.ToNNX(
#             _siglip.Module(
#                 num_classes=paligemma_config.width,
#                 variant="So400m/14",
#                 pool_type="none",
#                 scan=True,
#                 dtype_mm=config.dtype,
#             ))
#         img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
#         self.PaliGemma = nnx.Dict(llm=llm, img=img)
#         self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
#         self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
#         self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
#         self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
#         self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

#     @at.typecheck
#     def embed_prefix(
#         self, obs: _model.Observation
#     ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
#         input_mask = []
#         ar_mask = []
#         tokens = []
#         # embed images
#         for name in obs.images:
#             image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

#             tokens.append(image_tokens)
#             input_mask.append(einops.repeat(
#                 obs.image_masks[name],
#                 "b -> b s",
#                 s=image_tokens.shape[1],
#             ))
#             # image tokens attend to each other
#             ar_mask += [False] * image_tokens.shape[1]

#         # add language (aka tokenized inputs)
#         if obs.tokenized_prompt is not None:
#             tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
#             tokens.append(tokenized_inputs)
#             input_mask.append(obs.tokenized_prompt_mask)
#             # full attention between image and language inputs
#             ar_mask += [False] * tokenized_inputs.shape[1]
#         tokens = jnp.concatenate(tokens, axis=1)
#         input_mask = jnp.concatenate(input_mask, axis=1)
#         ar_mask = jnp.array(ar_mask)
#         return tokens, input_mask, ar_mask

#     @at.typecheck
#     def embed_suffix(
#         self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
#     ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
#         input_mask = []
#         ar_mask = []
#         tokens = []
#         # add a single state token
#         state_token = self.state_proj(obs.state)[:, None, :]
#         tokens.append(state_token)
#         input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
#         # image/language inputs do not attend to state or actions
#         ar_mask += [True]

#         # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
#         time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
#         # mix timestep + action information using an MLP
#         action_tokens = self.action_in_proj(noisy_actions)
#         time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
#         action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
#         action_time_tokens = self.action_time_mlp_in(action_time_tokens)
#         action_time_tokens = nnx.swish(action_time_tokens)
#         action_time_tokens = self.action_time_mlp_out(action_time_tokens)
#         tokens.append(action_time_tokens)
#         input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
#         # image/language/state inputs do not attend to action tokens
#         ar_mask += [True] + ([False] * (self.action_horizon - 1))
#         tokens = jnp.concatenate(tokens, axis=1)
#         input_mask = jnp.concatenate(input_mask, axis=1)
#         ar_mask = jnp.array(ar_mask)
#         return tokens, input_mask, ar_mask

#     @override
#     def compute_loss(self,
#                      rng: at.KeyArrayLike,
#                      observation: _model.Observation,
#                      actions: _model.Actions,
#                      *,
#                      train: bool = False) -> at.Float[at.Array, "*b ah"]:
#         preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
#         observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

#         batch_shape = actions.shape[:-2]
#         noise = jax.random.normal(noise_rng, actions.shape)
#         time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
#         time_expanded = time[..., None, None]
#         x_t = time_expanded * noise + (1 - time_expanded) * actions
#         u_t = noise - actions

#         # one big forward pass of prefix + suffix at once
#         prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
#         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
#         input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
#         ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
#         attn_mask = make_attn_mask(input_mask, ar_mask)
#         positions = jnp.cumsum(input_mask, axis=1) - 1
#         (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
#                                                          mask=attn_mask,
#                                                          positions=positions)
#         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

#         return jnp.mean(jnp.square(v_t - u_t), axis=-1)

#     @override
#     def sample_actions(
#         self,
#         rng: at.KeyArrayLike,
#         observation: _model.Observation,
#         *,
#         num_steps: int | at.Int[at.Array, ""] = 10,
#     ) -> _model.Actions:
#         observation = _model.preprocess_observation(None, observation, train=False)
#         # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
#         # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
#         dt = -1.0 / num_steps
#         batch_size = observation.state.shape[0]
#         noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

#         # first fill KV cache with a forward pass of the prefix
#         prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
#         prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
#         positions = jnp.cumsum(prefix_mask, axis=1) - 1
#         _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

#         def step(carry):
#             x_t, time = carry
#             suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t,
#                                                                            jnp.broadcast_to(time, batch_size))
#             # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
#             # other
#             suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
#             # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
#             # prefix tokens
#             prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
#             # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
#             # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
#             full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
#             assert full_attn_mask.shape == (
#                 batch_size,
#                 suffix_tokens.shape[1],
#                 prefix_tokens.shape[1] + suffix_tokens.shape[1],
#             )
#             # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
#             positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

#             (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
#                                                              mask=full_attn_mask,
#                                                              positions=positions,
#                                                              kv_cache=kv_cache)
#             assert prefix_out is None
#             v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

#             return x_t + dt * v_t, time + dt

#         def cond(carry):
#             x_t, time = carry
#             # robust to floating-point error
#             return time >= -dt / 2

#         x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
#         return x_0
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
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


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
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

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "... s"]]:
        input_mask = []
        ar_mask = []
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
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            if obs.token_ar_mask is None:
                ar_mask += [False] * tokenized_inputs.shape[1]
            else:
                ar_mask.extend([False] * tokenized_inputs.shape[1])
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        if obs.token_ar_mask is not None:
            token_ar_mask = obs.token_ar_mask.astype(jnp.bool_)
            image_prefix_len = tokens.shape[1] - obs.tokenized_prompt.shape[1]
            if token_ar_mask.ndim == 1:
                ar_mask = jnp.concatenate(
                    [jnp.zeros((image_prefix_len,), dtype=jnp.bool_), token_ar_mask], axis=0
                )
            else:
                ar_mask = jnp.concatenate(
                    [jnp.zeros((token_ar_mask.shape[0], image_prefix_len), dtype=jnp.bool_), token_ar_mask],
                    axis=1,
                )
        return tokens, input_mask, ar_mask

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
            # Subtask-only objective. Action tokens and action-side parameters do not
            # participate in this language-model pass.
            preprocess_rng = rng if train else None
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None], mask=attn_mask, positions=positions, adarms_cond=[None, None],
                deterministic=not train,
            )
            image_prefix_len = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
            text_out = prefix_out[:, image_prefix_len:-1]
            logits = self.PaliGemma.llm(pre_logits=text_out, method="logits")
            targets = observation.tokenized_prompt[:, 1:]
            loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            denom = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1.0)
            loss = -jnp.sum(selected * loss_mask, axis=-1) / denom
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
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def build_full_observation(self, observation: _model.Observation, subtask_tokens: jax.Array) -> _model.Observation:
        """Attach generated subtask tokens to the original task prefix."""
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Subtask generation requires tokenized_prompt and tokenized_prompt_mask")
        prefix = observation.tokenized_prompt
        prefix_mask = observation.tokenized_prompt_mask
        tokens = jnp.concatenate([prefix, subtask_tokens], axis=-1)
        mask = jnp.concatenate([prefix_mask, subtask_tokens != 0], axis=-1)
        return observation.replace(tokenized_prompt=tokens, tokenized_prompt_mask=mask, token_ar_mask=None, token_loss_mask=None)

    def generate_subtask(self, observation: _model.Observation, *, max_tokens: int = 50) -> jax.Array:
        """Greedy KV-cached decode, capped at 50 tokens by default.

        EOS (PaliGemma id 1) is retained as the final generated token; later
        positions are zero padded. If EOS is not reached, decoding stops at
        ``max_tokens``. Text detokenization stops before the first EOS.
        """
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Subtask generation requires a tokenized Task/Subtask prefix")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn, positions=prefix_positions, adarms_cond=[None, None]
        )
        image_prefix_len = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
        last_text_index = image_prefix_len + jnp.maximum(jnp.sum(observation.tokenized_prompt_mask, axis=1) - 1, 0)
        last_hidden = jnp.take_along_axis(prefix_out, last_text_index[:, None, None], axis=1)
        next_logits = self.PaliGemma.llm(pre_logits=last_hidden, method="logits")[:, 0]

        # Fixed-size cache permits a JAX loop while retaining per-token KV caching.
        cache_len = prefix_tokens.shape[1] + max_tokens
        def pad_layer_cache(x):
            pad_width = [(0, 0)] * x.ndim
            sequence_axis = 2 if x.ndim == 5 else 1
            pad_width[sequence_axis] = (0, cache_len - x.shape[sequence_axis])
            return jnp.pad(x, pad_width)

        cache = jax.tree.map(pad_layer_cache, cache)
        batch = prefix_tokens.shape[0]
        generated = jnp.zeros((batch, max_tokens), dtype=jnp.int32)
        finished = jnp.zeros((batch,), dtype=jnp.bool_)
        prefix_valid = prefix_mask

        def body(carry):
            i, logits, kv_cache, output, done = carry
            token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            token = jnp.where(done, 0, token)
            output = output.at[:, i].set(token)
            done = jnp.logical_or(done, token == 1)
            token_emb = self.PaliGemma.llm(token[:, None], method="embed")
            cache_position = jnp.full((batch, 1), prefix_tokens.shape[1] + i, dtype=jnp.int32)
            position = jnp.sum(prefix_valid, axis=1, keepdims=True).astype(jnp.int32) + i
            all_valid = jnp.concatenate(
                [prefix_valid, jnp.ones((batch, max_tokens), dtype=jnp.bool_)], axis=1
            )
            key_positions = jnp.arange(cache_len, dtype=jnp.int32)
            key_mask = all_valid & (key_positions[None, :] <= prefix_tokens.shape[1] + i)
            attention_mask = key_mask[:, None, :]
            (decoded, _), updated_cache = self.PaliGemma.llm(
                [token_emb, None], mask=attention_mask, positions=position, kv_cache=kv_cache,
                adarms_cond=[None, None], cache_positions=cache_position,
            )
            logits = self.PaliGemma.llm(pre_logits=decoded, method="logits")[:, 0]
            return i + 1, logits, updated_cache, output, done

        def condition(carry):
            i, _, _, _, done = carry
            return (i < max_tokens) & ~jnp.all(done)

        _, _, _, generated, _ = jax.lax.while_loop(
            condition, body, (jnp.asarray(0, dtype=jnp.int32), next_logits, cache, generated, finished)
        )
        return generated

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
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
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
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

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
