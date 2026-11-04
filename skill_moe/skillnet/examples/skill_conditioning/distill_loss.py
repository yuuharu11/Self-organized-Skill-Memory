"""Public ordinary flow objective, with batched autoregressive-mask compatibility.

The existing inference fork returns a [B,L] prefix mask, while the inherited
ordinary loss expects a 1-D prefix and concatenates on axis 0.
Only this concatenation is corrected here; corruption, target, balance term,
conditioning and model parameters remain the ordinary SkillNet objective.
"""

def joined_ar_mask(prefix,suffix,batch):
    import jax.numpy as jnp
    p=jnp.broadcast_to(prefix,(batch,prefix.shape[-1]))
    s=jnp.broadcast_to(suffix,(batch,suffix.shape[-1]))
    return jnp.concatenate([p,s],axis=1)


def ordinary_flow_loss(model,rng,observation,actions,*,train):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0_moe_skill import make_attn_mask,_apply_skill_embed_override
    if observation.memory_features is not None:raise ValueError('retrieval anchors forbidden')
    prep_rng,noise_rng,time_rng=jax.random.split(rng,3)
    observation=model_lib.preprocess_observation(prep_rng,observation,train=train)
    noise=jax.random.normal(noise_rng,actions.shape)
    time=jax.random.beta(time_rng,1.5,1,actions.shape[:-2])*.999+.001
    x_t=time[...,None,None]*noise+(1-time[...,None,None])*actions
    target=noise-actions
    prefix,prefix_mask,prefix_ar=model.embed_prefix(observation)
    suffix,suffix_mask,suffix_ar,adarms_cond=model.embed_suffix(observation,x_t,time)
    valid=jnp.concatenate([prefix_mask,suffix_mask],axis=1)
    mask=make_attn_mask(valid,joined_ar_mask(prefix_ar,suffix_ar,actions.shape[0]))
    positions=jnp.cumsum(valid,axis=1)-1
    skills=model.PaliGemma.llm(observation.skills,method='skill_embed')
    skills=_apply_skill_embed_override(skills,observation.skill_embed_override)
    if model.positional_encoding:
        positions_=jnp.broadcast_to(jnp.arange(observation.skills.shape[1])[None,:],observation.skills.shape)
        skills=jnp.concatenate([skills,model.PaliGemma.llm(positions_,method='position_embed')],axis=-1)
    (_,output),_,balance=model.PaliGemma.llm([prefix,suffix],skill_embeds=skills,skill_mask=observation.skill_mask,mask=mask,positions=positions,adarms_cond=[None,adarms_cond])
    pred=model.action_out_proj(output[:,-model.action_horizon:])
    return jnp.mean(jnp.square(pred-target),axis=-1),balance
