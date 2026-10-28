import numpy as np
import pytest

from skill_moe.skillnet.examples.emergent.light.methods.fsq import FSQ


@pytest.mark.parametrize("codes", [16, 32, 64])
def test_fsq_fit_encode_decode_prior_and_memory(codes):
    rng=np.random.default_rng(17); n=40; h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),
           "state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),
           "key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),
           "init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":4,"epochs":1,"batch_size":16}
    m=FSQ(n_codes=codes,epochs=1,batch_size=16,width=32); m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4])
    logits,rp=m.prior(train["feature"][:4],train["state"][:4])
    y=m.decode(m.base_features(b),r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,) and b.min()>=0 and b.max()<codes
    assert r.shape==(4,4) and logits.shape==(4,codes) and rp.shape==(4,4)
    assert y.shape==(4,h,7) and np.isfinite(y).all()
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    assert m.blend_base(candidates,weights).shape==(m.base_dim,)
    assert m.describe()["neural_weights_fixed_on_memory_add"]
