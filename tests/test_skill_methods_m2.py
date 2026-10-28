import numpy as np

from skill_moe.skillnet.examples.emergent.light.methods.m2 import M2


def test_m2_fit_encode_decode_prior_and_memory():
    rng=np.random.default_rng(10);n=48;h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),"state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),"key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),"init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":3,"epochs":1,"batch_size":16}
    m=M2(n_codes=16,epochs=1,batch_size=16,width=32);m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4]);bd,pr=m.prior(train["feature"][:4],train["state"][:4])
    out=m.decode(m.base_features(b),r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,) and r.shape==(4,4) and bd.shape==(4,16) and pr.shape==(4,4)
    assert out.shape==(4,h,7) and np.isfinite(out).all()
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    base=m.blend_base(candidates,weights);assert base.shape==(16,) and np.isclose(weights.sum(),1)
    assert m.memory_add(train["key"][1],b[1],{"episode_id":"e1","init_state_id":"i1"})[1] in {"added_novel","near_existing"}
    assert m.describe()["base"].startswith("EMA VQ code with 16")
