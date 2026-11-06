import numpy as np
import json

from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.data import (
    training_row_mask,
)
from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.representations import (
    apply_pose, canonicalize, decode_canonical, decode_original, model_code, m6_fit_generic, medoid_generic, rotation_to_x,
    decode_from_bundle, pca_fit, pca_encode,
    raw_pca_input,
)
from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.run import (
    _FrozenM1Decoder, conditioning_row_for_segment, first_row_in_segment, load_pose_rows,
    make_rep_models, make_row_lookup, save_model_bundle, ensure_query_segment_coverage,
)


def test_rotation_happens_in_physical_delta_space_and_roundtrips():
    q01 = np.array([-1, -1, -1, -2, -2, -2, -1], np.float32)
    q99 = np.array([1, 1, 1, 2, 2, 2, 1], np.float32)
    # 90 degree direction in physical xyz, with an independent rotation delta.
    physical = np.tile(np.array([0, .1, 0, .02, 0, 0, -1], np.float32), (12, 1))
    normed = (physical - q01) / (q99 - q01) * 2 - 1
    restored_input = (normed + 1) * .5 * (q99 - q01) + q01
    canonical, pose = canonicalize(restored_input)
    direction = rotation_to_x(np.array([0, 1, 0], np.float32)) @ np.array([0, 1, 0])
    np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-6)
    restored = apply_pose(canonical, pose)
    np.testing.assert_allclose(restored, physical[:, :6], atol=1e-6)
    ref=np.stack([np.linspace(0,1,32),np.zeros(32),np.zeros(32)],axis=1).astype(np.float32)
    can_p,pose_p=canonicalize(physical,procrustes_ref=ref)
    np.testing.assert_allclose(apply_pose(can_p,pose_p),physical[:,:6],atol=1e-6)
    # Grip is carried separately and never rotated.
    np.testing.assert_array_equal(physical[:, 6], np.full(12, -1, np.float32))


def test_uncompressed_canonical_shape_roundtrips_without_code_projection(tmp_path):
    rng=np.random.default_rng(41)
    physical=np.column_stack([rng.normal(size=(32,6)).astype(np.float32)*.03,
                              np.where(np.arange(32)<16,-1.,1.).astype(np.float32)])
    model={"kind":"canonical_raw"}
    code,pose,_=model_code(model,physical)
    assert code.shape==(32*6,)
    can,grip=decode_canonical(model,code,pose,32)
    np.testing.assert_allclose(np.column_stack([apply_pose(can,pose),grip]),physical,atol=2e-6)
    q01=np.full(7,-1.,np.float32);q99=np.full(7,1.,np.float32)
    meta=save_model_bundle(tmp_path,model,q01,q99)
    (tmp_path/'manifest.json').write_text(json.dumps({**meta,'q01':q01.tolist(),'q99':q99.tolist()}))
    normalized=decode_from_bundle(tmp_path,code[None],{k:np.asarray(v)[None] if np.asarray(v).ndim else v for k,v in pose.items()},32)
    expected=physical
    np.testing.assert_allclose(normalized[0],expected,atol=2e-5)


def test_general_minimal_rotation_is_proper_orthogonal_and_maps_to_x():
    v=np.asarray([.31,-.47,.82],np.float64)
    v=v/np.linalg.norm(v)
    R=rotation_to_x(v)
    np.testing.assert_allclose(R@v,[1.,0.,0.],atol=2e-6)
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=2e-6)
    np.testing.assert_allclose(np.linalg.det(R),1.,atol=2e-6)


def test_raw_pca_uses_native_fixed_window_and_normalizes_before_fit():
    q01=np.asarray([-2.,-3.,-4.,-5.,-6.,-7.,-8.],np.float32)
    q99=np.asarray([2.,3.,4.,5.,6.,7.,8.],np.float32)
    physical=np.arange(70,dtype=np.float32).reshape(10,7)/20
    expected=(physical-q01)/(q99-q01+1e-6)*2-1
    native=raw_pca_input(physical,q01,q99,native_length=10)
    phase=raw_pca_input(physical,q01,q99)
    assert native.shape==(10,7)
    assert native.reshape(-1).shape==(70,)
    assert phase.shape==(32,7)
    np.testing.assert_allclose(native,expected,atol=1e-7,rtol=0)
    with np.testing.assert_raises(ValueError):
        raw_pca_input(physical[:9],q01,q99,native_length=10)


def test_fixed10_raw_pca_fit_and_encode_use_native_70_values():
    rng=np.random.default_rng(52)
    q01=np.full(7,-1.,np.float32);q99=np.full(7,1.,np.float32)
    segments=[{"a":rng.normal(size=(10,7)).astype(np.float32)*.1} for _ in range(40)]
    models=make_rep_models(segments,{},np.zeros((40,8),np.float32),np.arange(40),
                           segmentation_name="S0_fixed10",q01=q01,q99=q99)
    model=next(m for m in models if m["kind"]=="pca_raw" and m["k"]==24)
    assert model["pca_input_length"]==10
    assert model["pca"]["mean"].shape==(70,)
    assert model["pca"]["components"].shape==(24,70)
    expected=np.stack([raw_pca_input(s["a"],q01,q99,native_length=10).reshape(-1) for s in segments])
    np.testing.assert_allclose(model["pca"]["mean"],expected.mean(0),atol=2e-7,rtol=0)
    code,_,_=model_code(model,segments[0]["a"])
    np.testing.assert_allclose(code,pca_encode(model["pca"],expected[0]),atol=2e-6,rtol=0)


def test_global_dataset_row_ids_are_not_pool_positions():
    rows = {"task": np.array([0, 12, 12, 12]), "episode": np.array([0, 7, 7, 7]),
            "frame": np.array([0, 0, 10, 20])}
    lookup = make_row_lookup(rows)
    seg = {"task": 12, "episode": 7, "start": 10, "length": 20, "row_id": 4001}
    assert first_row_in_segment(seg, lookup) == 2  # dataset row ID, not memory-pool offset 1


def test_query_only_fallback_covers_rows_outside_variable_segments():
    rows={"task":np.array([4]),"episode":np.array([9]),"frame":np.array([25])}
    actions=np.arange(40*7,dtype=np.float32).reshape(40,7)
    episodes={(4,9):{"frame":np.arange(40),"actions_phys":actions,"meta":{"present":True}}}
    got,n=ensure_query_segment_coverage({},[0],rows,episodes,window=10)
    assert n.tolist()==[0]
    seg=got[(4,9)][0]
    assert (seg["start"],seg["length"],seg["row_id"])==(25,10,0)
    np.testing.assert_array_equal(seg["a"],actions[25:35])


def test_variable_segment_m1_context_falls_back_to_nearest_prior_global_row():
    rows={"task":np.array([12,12]),"episode":np.array([7,7]),"frame":np.array([0,20])}
    lookup=make_row_lookup(rows)
    seg={"task":12,"episode":7,"start":10,"length":10,"row_id":-1}
    assert conditioning_row_for_segment(seg,lookup)==0


def test_compressed_pose_bundle_is_read_once_and_split_by_row(tmp_path):
    p=tmp_path/"pose.npz"
    np.savez_compressed(p,scale=np.array([1.,2.,3.],np.float32),
                        rotation=np.broadcast_to(np.eye(3,dtype=np.float32),(3,3,3)),
                        q01=np.arange(7,dtype=np.float32))
    got=load_pose_rows(p,3)
    np.testing.assert_array_equal([r["scale"] for r in got],[1.,2.,3.])
    np.testing.assert_array_equal(got[1]["rotation"],np.eye(3,dtype=np.float32))
    np.testing.assert_array_equal(got[2]["q01"],np.arange(7,dtype=np.float32))


def test_fit_rows_exclude_long_heldout_memory_and_dev():
    rows = {"task": np.array([0, 11, 17, 12, 13]),
            "memory": np.array([False, False, False, True, False]),
            "validation": np.array([False, False, False, False, True])}
    np.testing.assert_array_equal(training_row_mask(rows), [False, True, False, False, False])


def test_dp_threshold_falls_back_to_minimum_windows_when_no_segment_can_meet_epsilon():
    from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.data import threshold_segments
    # Alternating large commands ensure every 10-step constant approximation
    # is above epsilon; an episode-wide segment must not be mislabeled valid.
    x=np.asarray([(-1. if i%2 else 1.)*np.ones(7,np.float32) for i in range(30)])
    seq={"actions_norm":x,"frame":np.arange(len(x))}
    got=threshold_segments(seq,.005,"dp",minimum=10)
    assert got==[(0,10),(10,20),(20,30)]


def test_generic_m6_is_variable_dimension_and_medoid_is_member():
    x = np.array([[0., 0.], [.01, 0.], [2., 2.], [2.01, 2.]], np.float32)
    fitted = m6_fit_generic(x, [1, 1, 2, 2], [1, 2, 3, 4], kappa=.5,
                            min_support=2, min_episodes=2, seed=4)
    assert fitted["row_units"].shape == (4,)
    for members in fitted["members_local"]:
        assert medoid_generic(members, x) in members


def test_artifact_decoder_returns_official_normalized_action(tmp_path):
    rng=np.random.default_rng(2)
    q01=np.array([-1,-1,-1,-1,-1,-1,-1],np.float32)
    q99=-q01
    raw=rng.normal(size=(30,10*7)).astype(np.float32)*.1
    pca=pca_fit(raw,8)
    model={"kind":"pca_raw","pca":pca,"k":8,"code_dim":8,
           "pca_input_length":10,"q01":q01,"q99":q99}
    meta=save_model_bundle(tmp_path,model,q01,q99)
    (tmp_path/"manifest.json").write_text(json.dumps({**meta,"q01":q01.tolist(),"q99":q99.tolist()}))
    action=rng.normal(size=(10,7)).astype(np.float32)*.1
    code=pca_encode(pca,action.reshape(-1))[None]
    decoded=decode_from_bundle(tmp_path,code,{},10)
    assert decoded.shape==(1,10,7)
    assert decoded.dtype==np.float32 and np.isfinite(decoded).all()


def test_m1_artifact_decoder_matches_fitted_codec():
    import torch
    from skill_moe.skillnet.examples.emergent.light.methods.m1 import M1
    torch.set_num_threads(8)
    rng=np.random.default_rng(3)
    n=12
    a=rng.normal(size=(n,10,32)).astype(np.float32)
    s=rng.normal(size=(n,8)).astype(np.float32)
    f=rng.normal(size=(n,16)).astype(np.float32)
    train={"actions":a,"state":s,"feature":f,"key":np.zeros((n,8),np.float32),
           "episode_id":np.array([f"ep{i//2}" for i in range(n)]),
           "init_state_id":np.array([f"ep{i//2}" for i in range(n)]),"seed":7}
    model=M1(epochs=1,batch_size=6,seed_value=7,base_dim=4,residual_dim=2,width=16,device="cpu")
    model.fit(train)
    b,r=model.encode(a[:2],s[:2])
    expected=model.decode(b,r,f[:2],s[:2])
    payload={"stats":{k:torch.tensor(getattr(model,k)) for k in ("am","ass","sm","ss","fm","fs")},
             "nets":{k:getattr(model,k).state_dict() for k in ("base_decoder","residual_decoder")}}
    actual=_FrozenM1Decoder(payload).decode(b,r,f[:2],s[:2])
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
