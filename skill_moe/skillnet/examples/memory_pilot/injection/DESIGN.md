# Memory-to-action injection design

This package defines the adapter independently of checkpoint loading. The
`Pi0` model has an explicit `attach_memory_injection(adapter)` method; attach
the adapter after restoring the released checkpoint and before constructing a
`Policy` (which JIT-wraps the model). The adapter returns residuals and does not
own or mutate base-model activations.

## Adapter choices

`MemoryInjection` has three modes:

- `action_tokens`: pooled memory `c: [B, Dz]` passes through a two-layer MLP
  and returns a zero-start residual `[B, H, width]` (broadcast across the
  requested horizon). Add it to the action token embeddings after action and time inputs have been combined, before they
  enter the action expert.
- `adarms_cond`: the same pooled-vector MLP returns `[B, width]`, to add to
  `adarms_cond` after its time MLP in `embed_suffix` and before `RMSNorm` consumes
  it in the action-expert blocks. Since RMSNorm maps this condition into scale,
  shift, and gate values, injecting before that projection preserves the exact
  existing modulation when the residual is zero.
- `candidate_attention`: action tokens `[B, H, width]` query candidate memory
  `[B, K, Dz]` with mask `[B, K]`. The output is a residual `[B, H, width]`.
  An exactly zero scalar gate (through `tanh`) gates the cross-attention output;
  an all-masked set returns exact zero both at initialization and after training
  (including when the output projection bias has moved).

For pooled modes, `condition_out` has an exactly zero kernel and bias. For
candidate attention, the attention projections are normally initialized and
the scalar gate is zero. The caller casts every residual to the base activation
dtype before adding it, so a zero fp32 adapter output does not promote a bf16
activation and alter the model's numerical path. Thus all modes return exact
zero residuals at initialization.

Zero-start learning is staged. On the first pooled-mode update, the final
`condition_out` projection receives gradients while `condition_in` receives
zero; after `condition_out` moves from zero, gradients reach `condition_in`.
For candidate attention, the gate receives a gradient initially (provided at
least one candidate is valid); query/key/value and output-projection gradients
are zero behind the zero gate, then become available after the gate moves.
Do not also zero-initialize `context_out`: with both the gate and output
projection at zero, the gate gradient would also be zero.

## Observation payload and integration

Pass the retrieved representation in new optional observation fields, separate
from the existing skill-ID path:

- `memory_features: [B, Dz]` for a pooled representation, or
- `memory_candidates: [B, K, Dz]` and `memory_candidate_mask: [B, K]` for a
  candidate set.

The inference data transform should preserve these fields as it does other
model inputs. `preprocess_observation` must carry them through unchanged. In
`Pi0.sample_actions`, the prefix is embedded once and its KV cache is reused;
`Pi0.embed_suffix` is called anew at every diffusion step. The follow-up
connection should read the same per-observation memory payload there and add
the selected residual before the suffix enters the action expert. It must not
modify the cached prefix tokens/KV. For `action_tokens`, the integration point
is after the action/time combination. For `adarms_cond`, it is after the time
MLP and before RMSNorm's scale/shift/gate projection. For
`candidate_attention`, use the current suffix action tokens as queries and the
unchanged retrieved candidate set as keys/values on every diffusion step; the
query-dependent attended residual can change with the denoising state even
though the memory condition is available at every step. An explicit
`memory_enabled` switch should bypass the adapter and residual add entirely
when false.

The existing `skill_embed_override` is the G1 hook for replacing the skill
embedding vectors. It is not a memory payload and should remain independent.
The action-side adapter can therefore be evaluated with skills unchanged and
memory disabled/enabled as a controlled pair.

## Frozen and trainable parameters

Freeze all released SkillNet parameters: visual encoder, language model,
action expert, skill embedder, and action/time projections. Train only the
selected `MemoryInjection` instance. For the NNX pilot, keep the base model out
of the optimizer and build the optimizer on the adapter module only:

```python
from flax import nnx
import optax

adapter_optimizer = nnx.Optimizer(adapter, optax.adam(learning_rate), wrt=nnx.Param)
adapter_grads = nnx.grad(adapter_loss)(adapter)  # adapter_loss closes over the frozen policy
adapter_optimizer.update(adapter_grads)
```

This leaves released model parameters outside the optimizer. Do not instantiate
all modes at once for one run; compare the modes in separate runs or explicitly
configured ablations.

For the published action expert `width=1024`, default `hidden_dim=256`,
`attention_dim=128`, and a representative `Dz=128`:

| Mode | Trainable parameters |
| --- | ---: |
| `action_tokens` | 296,192 |
| `adarms_cond` | 296,192 |
| `candidate_attention` | 296,321 |

Counts include linear biases and the scalar gate. Formulas for arbitrary sizes:

- Pooled: `Dz*hidden_dim + hidden_dim + hidden_dim*width + width`.
- Candidate attention: `(width*A + A) + 2*(Dz*A + A) + (A*width + width) + 1`,
  where `A=attention_dim`.

## Checks

`tests/test_memory_pilot_injection_adapter.py` verifies bf16 identity at
zero-start, all-masked neutrality after training, and staged gradients through
the pooled MLP and candidate-attention gate using dummy inputs. This change does
not alter model files or pretrained parameters.
