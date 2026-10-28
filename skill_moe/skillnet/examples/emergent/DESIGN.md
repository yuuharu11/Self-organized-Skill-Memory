# Emergent z training configs

This implementation follows [the emergent-skill plan](../../../../docs/research/emergent_skill_plan.md)
and [verification protocol, §§1 and 5](../../../../docs/research/emergent_skill_verification.md).

## Registered configs

| Config | Initialization | Skill IDs | z injection | Horizon |
|---|---|---:|---|---:|
| `pi05_libero90_emergent_z` | pi0.5 base | no | router + one suffix token | 25 |
| `pi05_libero90_emergent_z_router_only` | pi0.5 base | no | router only | 25 |
| `pi05_libero90_emergent_z_h10` | pi0.5 base | no | router + one suffix token | 10 |
| `pi05_libero90_skillid_z` | SkillNet checkpoint | yes | router + one suffix token | 25 |

All four freeze PaliGemma and train the action expert, posterior encoder, observation prior,
and z projections. The objective is flow-matching reconstruction plus
`balance_loss_weight * balance_loss + kl_beta * KL(q || p)`. All registered configs use
`balance_loss_weight=1.0`; each MoEFeedForward applies `router_loss_scale=0.01` to its
load-balancing loss, matching SkillNet. Training data defaults to
`/work/.skillnet-state/data/lerobot/emergent/libero_90_unsegmented`, with LIBERO-90 assets
under its `assets/` directory. Override paths with `SKILLNET_EMERGENT_DATASET_ROOT`,
`SKILLNET_EMERGENT_ASSETS_DIR`, `SKILLNET_EMERGENT_ASSET_ID`, and `SKILLNET_EMERGENT_REPO_ID`.

## Per-chunk inputs and outputs

For a batch of observations, images and tokenized prompt are encoded by frozen PaliGemma;
proprioception is padded to `s: [B, 32]`. Dataset actions are clean chunks `A: [B, H, 7]`;
the existing `PadStatesAndActions` transform pads them to `[B, H, 32]` for the model.
The posterior encoder maps `(A, s)` to diagonal-Gaussian parameters
`(mu_q, logvar_q): [B, Dz]`; the observation prior maps pooled VLM features and `s` to
`(mu_p, logvar_p): [B, Dz]`. Training samples `z ~ q`; inference uses the prior mean unless
an external z is supplied.

The router projection produces four key/value tokens `[B, 4, Dkey]`. The token projection
produces one action-expert suffix token `[B, 1, Daction]`; for pi0.5 it precedes the H
action tokens in the same causal block. The decoder predicts the padded chunk `[B, H, 32]`;
the first 7 entries are the LIBERO action and the remaining 25 entries are zero-padded training targets.
With `z_inject="token"` and no skill IDs, the router receives one zero key with its mask off;
the all-masked cross-attention contributes zero, so expert routing is uniform and z affects
the suffix token only. With router injection, z supplies the router's key/value tokens.

The load-balancing auxiliary returned by each MoE layer uses `router_loss_scale=0.01`;
the trainer weights it by `balance_loss_weight=1.0`. Gating diagnostics record the fraction
of action-expert suffix tokens assigned top-1 to each expert, separately for every layer,
at the final diffusion step.
