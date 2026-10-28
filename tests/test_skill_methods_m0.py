import numpy as np

from skill_moe.skillnet.examples.emergent.light.methods.m0 import M0


def test_m0_fit_encode_decode_prior_and_memory():
    rng=np.random.default_rng(8);n=32;h=5
    train={"actions":rng.normal(size=(n,h,7)).astype(np.float32),"state":rng.normal(size=(n,8)).astype(np.float32),
           "feature":rng.normal(size=(n,24)).astype(np.float32),"key":rng.normal(size=(n,10)).astype(np.float32),
           "episode_id":np.asarray([f"e{i}" for i in range(n)]),"init_state_id":np.asarray([f"i{i}" for i in range(n)]),
           "seed":1,"epochs":1,"batch_size":16}
    m=M0(epochs=1,batch_size=16,width=32);m.fit(train)
    b,r=m.encode(train["actions"][:4],train["state"][:4]);pb,pr=m.prior(train["feature"][:4],train["state"][:4])
    out=m.decode(b,r,train["feature"][:4],train["state"][:4])
    assert b.shape==(4,32) and r is None and pr is None and out.shape==(4,h,7)
    assert np.isfinite(out).all() and pb.shape==(4,32)
    assert m.memory_add(train["key"][0],b[0],{"episode_id":"e0","init_state_id":"i0","seed":True})[0]
    candidates,weights,_=m.memory_query(train["key"][1],{"episode_id":"q","init_state_id":"qi"})
    base=m.blend_base(candidates,weights)
    assert candidates.shape[1:]==(32,) and np.isclose(weights.sum(),1)
    assert m.memory_add(train["key"][1],b[1],{"episode_id":"e1","init_state_id":"i1"})[1] in {"added_novel","near_existing"}
    assert m.describe()["name"]=="M0"
