"""Shape skills (PCA, AE, VQ) and open-ended raw skill memories."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.data import (
    training_row_mask,
)
from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.representations import (
    apply_pose,
    canonicalize,
    decode_canonical,
    decode_from_bundle,
    m6_fit_generic,
    medoid_generic,
    model_code,
    pca_encode,
    pca_fit,
    raw_pca_input,
    rotation_to_x,
)
from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.run import (
    _FrozenM1Decoder,
    conditioning_row_for_segment,
    ensure_query_segment_coverage,
    first_row_in_segment,
    load_pose_rows,
    make_rep_models,
    make_row_lookup,
    save_model_bundle,
)
from skill_moe.skillnet.examples.memory_pilot.pi05_memory.skill_memories.build import (
    RawFunctional,
    assert_no_long,
    gap_threshold,
    m6_fit,
    unit_medoid,
)
from tests.support import (
    PI05,
    load_script,
)

ae = load_script(PI05 / "shape_skills/ae.py")
vq = load_script(PI05 / "shape_skills/vq.py")


# ── Shape-skill representations ───────────────────────────────────────────────────────────────────

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
    from skill_moe.skillnet.examples.memory_pilot.pi05_memory.shape_skills.data import (
        threshold_segments,
    )
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


# ── Autoencoder ───────────────────────────────────────────────────────────────────────────────────

def test_masked_mse_ignores_padding_values():
    torch.set_num_threads(1)
    y=torch.tensor([[[1.,2.],[3.,4.]]]);p=y.clone();m=torch.tensor([[True,True]])
    expected=ae.masked_mse(p,y,m)
    yp=torch.cat([y,torch.full((1,4,2),100.)],1)
    pp=torch.cat([p,torch.full((1,4,2),-50.)],1)
    mp=torch.tensor([[True,True,False,False,False,False]])
    torch.testing.assert_close(expected,ae.masked_mse(pp,yp,mp))


def test_pad_batch_uses_global_lmax_and_zero_suffix_mask():
    x=[np.ones((3,7),np.float32),np.full((5,7),2.,np.float32)]
    y=[v.copy() for v in x]
    xb,yb,phase,mask,length=ae._pad_batch(x,y,np.zeros(7),np.ones(7),np.zeros(7),np.ones(7),8,'cpu')
    assert tuple(xb.shape)==(2,8,7)
    assert mask.tolist()==[[True,True,True,False,False,False,False,False],
                           [True,True,True,True,True,False,False,False]]
    assert torch.count_nonzero(xb[~mask])==0 and torch.count_nonzero(yb[~mask])==0
    assert length.shape==(2,1) and phase.shape==(2,8)


def test_encoder_latent_is_independent_of_arbitrary_padding_values():
    torch.set_num_threads(1);torch.manual_seed(2)
    _,_,_,Model=ae._torch_classes();model=Model(latent_dim=8).eval()
    x=torch.randn(2,7,7);phase=torch.zeros(2,7);mask=torch.tensor([[1,1,1,0,0,0,0],[1,1,1,1,1,0,0]],dtype=torch.bool)
    lengths=torch.tensor([[.4],[.7]])
    x2=x.clone();x2[~mask]=987.
    with torch.no_grad():z0=model.encode(x,phase,mask,lengths);z1=model.encode(x2,phase,mask,lengths)
    torch.testing.assert_close(z0,z1)


def test_length_is_an_explicit_encoder_and_decoder_input():
    torch.set_num_threads(1);torch.manual_seed(3)
    _,_,_,Model=ae._torch_classes();model=Model(latent_dim=8).eval()
    x=torch.zeros(1,5,7);phase=torch.linspace(0,1,5)[None];mask=torch.ones(1,5,dtype=torch.bool)
    zshort=model.encode(x,phase,mask,torch.tensor([[.2]]))
    zlong=model.encode(x,phase,mask,torch.tensor([[.9]]))
    assert zshort.shape==(1,8) and not torch.allclose(zshort,zlong)
    assert model.decode(zshort,phase,torch.tensor([[.2]])).shape==(1,5,7)


def test_physical_augmentation_rotates_position_and_rotation_but_not_gripper():
    x=np.arange(70,dtype=np.float32).reshape(10,7)
    y,pose=ae.augment_physical(x,np.random.default_rng(4),scale_range=(1.2,1.2))
    np.testing.assert_allclose(y[:,6],x[:,6])
    np.testing.assert_allclose(pose['rotation'].T@pose['rotation'],np.eye(3),atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(y[:,:3]),np.linalg.norm(x[:,:3])*1.2,rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(y[:,3:6]),np.linalg.norm(x[:,3:6])*1.2,rtol=1e-5)


def test_canonical_shape_keeps_gripper_as_seventh_channel():
    x=np.zeros((13,7),np.float32)
    x[:,0]=.01;x[:,3]=.02;x[:,6]=np.linspace(-1,1,13)
    ref=np.cumsum(np.tile([1.,0.,0.],(32,1)),axis=0)
    y,pose=ae.canonicalize_7(x,ref)
    assert y.shape==(13,7)
    np.testing.assert_allclose(y[:,6],x[:,6])
    assert pose['rotation'].shape==(3,3)


def test_lmax_is_ceiling_of_nonlong_p99_statistic():
    l,p=ae.choose_lmax([10,20,30,40,100],percentile=.8)
    np.testing.assert_allclose(p,52.)
    assert l==52


# ── VQ ────────────────────────────────────────────────────────────────────────────────────────────

def test_padding_is_masked_and_does_not_change_valid_frame_reconstruction_loss():
    torch.set_num_threads(1)
    x = torch.tensor([[[1., 2.], [3., 4.]]])
    y = torch.tensor([[[1., 2.], [3., 4.]]])
    mask = torch.tensor([[True, True]])
    base = vq._masked_reconstruction_loss(y, x, mask)
    xp = torch.cat([x, torch.zeros((1, 3, 2))], dim=1)
    yp = torch.cat([y, torch.full((1, 3, 2), 100.)], dim=1)
    mp = torch.tensor([[True, True, False, False, False]])
    padded = vq._masked_reconstruction_loss(yp, xp, mp)
    torch.testing.assert_close(base, padded)


def test_pad_sequences_tracks_valid_lengths_and_mask_only():
    x, phase, mask, lengths = vq.pad_sequences([np.ones((2, 7)), np.ones((5, 7))*3])
    assert x.shape == (2, 5, 7)
    np.testing.assert_array_equal(lengths.numpy(), [2, 5])
    np.testing.assert_array_equal(mask.numpy().sum(1), [2, 5])
    np.testing.assert_array_equal(x.numpy()[0, 2:], 0.)
    np.testing.assert_allclose(phase.numpy()[0, :2], [0., 1.])


def test_vqvae_forward_handles_masked_variable_lengths_and_all_residual_branches():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    _, _, _, _, Model = vq._torch_classes()
    model = Model(input_dim=7, latent_dim=8, k=16, residual_dims=(4, 8))
    x, phase, mask, _ = vq.pad_sequences([np.random.randn(6, 7).astype(np.float32),
                                         np.random.randn(11, 7).astype(np.float32)])
    pred, ids, qloss, reg = model(x, phase, mask, update=True)
    assert pred['0'].shape == (2, 11, 7)
    assert pred['4'].shape == pred['8'].shape
    assert ids.shape == (2,) and torch.isfinite(qloss) and torch.isfinite(reg)
    losses = vq._masked_reconstruction_loss(pred['0'], x, mask)
    assert torch.isfinite(losses).all()


def test_masked_padding_cannot_change_the_segment_latent():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    _, _, _, _, Model = vq._torch_classes()
    model = Model(input_dim=7, latent_dim=8, k=16, residual_dims=(4, 8)).eval()
    x, phase, mask, _ = vq.pad_sequences([np.random.default_rng(7).normal(size=(5, 7)).astype(np.float32)])
    x_changed = x.clone()
    x_changed[:, 5:] = 123.0  # invalid padding is deliberately nonzero
    with torch.no_grad():
        z0, *_ = model.encode_quantized(x, phase, mask, update=False)
        z1, *_ = model.encode_quantized(x_changed, phase, mask, update=False)
    torch.testing.assert_close(z0, z1)


def test_fit_split_excludes_long_and_heldout_tasks():
    segments = [{'task': 0}, {'task': 16}, {'task': 17}, {'task': 39}]
    ids = vq._training_indices(segments)
    np.testing.assert_array_equal(ids, [1, 3])


def test_procrustes_transform_accepts_variable_length_with_train_reference():
    rng=np.random.default_rng(3)
    seq=rng.normal(0,.01,(17,7)).astype(np.float32)
    seq[:,6]=np.where(np.arange(len(seq))<8,-1.,1.)
    ref=np.cumsum(rng.normal(0,.01,(32,3)).astype(np.float32),axis=0)
    x,pose=vq._seq_input(seq,'procrustes',np.full(7,-1,np.float32),np.full(7,1,np.float32),ref)
    assert x.shape==(17,7) and pose['rotation'].shape==(3,3)
    np.testing.assert_allclose(pose['rotation'].T@pose['rotation'],np.eye(3),atol=1e-5)


# ── Raw skill memories — CPU-only checks for open-ended M6-style raw action memories. ─────────────

def test_m6_registers_supported_novel_regions_without_fixed_k():
    rng = np.random.default_rng(31)
    # Two separated functional regions, sufficiently supported over episodes.
    x = np.concatenate([
        rng.normal(0.0, 0.005, (24, 70)),
        rng.normal(0.8, 0.005, (24, 70)),
    ]).astype(np.float32)
    row_ids = np.arange(len(x), dtype=np.int64)
    tasks = np.full(len(x), 12, np.int32)
    episodes = np.arange(len(x), dtype=np.int32)
    result = m6_fit(x, row_ids, tasks, episodes, RawFunctional(), kappa=.1,
                    epsilon_override=.01, seed=4, pending_batch=8)
    assert result['registered'] == 2
    assert np.all(result['row_units'] >= 0)
    assert sorted(result['support'].tolist()) == [24, 24]
    assert all(len(es) == 24 for es in result['episode_sets'])
    assert len(result['centers']) == len(result['members_local'])


def test_medoid_is_an_actual_member_minimizing_mean_functional_distance():
    x = np.asarray([[0.0] * 70, [0.01] * 70, [0.04] * 70, [1.0] * 70], np.float32)
    members = np.asarray([0, 1, 2], np.int64)
    medoid = unit_medoid(members, x, RawFunctional())
    metric = RawFunctional()
    costs = [np.mean([metric.distance(x[i], x[j]) for j in members]) for i in members]
    assert medoid in members
    assert costs[list(members).index(medoid)] == min(costs)


def test_no_long_guard_rejects_long_rows_and_eval_overlap():
    rows = {'task': np.asarray([0, 11, 12, 2]), 'memory': np.asarray([1, 1, 1, 0])}
    eval_ids = np.asarray([3], np.int64)
    assert_no_long([1, 2], eval_ids, rows, context='test')
    try:
        assert_no_long([0], eval_ids, rows, context='test')
    except AssertionError as exc:
        assert 'Long task rows leaked' in str(exc)
    else:
        raise AssertionError('Long memory row was not rejected')
    try:
        assert_no_long([3], eval_ids, rows, context='test')
    except AssertionError as exc:
        assert 'Long Eval rows leaked' in str(exc)
    else:
        raise AssertionError('Eval row was not rejected')


def test_gap_threshold_prefers_corrected_over_summary(tmp_path):
    (tmp_path / 'summary_corrected.json').write_text(json.dumps({'threshold': 0.02}))
    (tmp_path / 'summary.json').write_text(json.dumps({'threshold': 0.05}))
    assert gap_threshold(tmp_path) == 0.02


def test_gap_threshold_falls_back_to_summary(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps({'threshold': 0.01549410422449}))
    assert gap_threshold(tmp_path) == 0.01549410422449


def test_gap_threshold_missing_both_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='summary_corrected.json'):
        gap_threshold(tmp_path)
