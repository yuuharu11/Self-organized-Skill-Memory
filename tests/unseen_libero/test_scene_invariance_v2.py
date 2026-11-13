import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


MODULE = Path(__file__).parents[2] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/scene_invariance_v2.py"
spec = importlib.util.spec_from_file_location("scene_invariance_v2_test", MODULE)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def test_action_type_split_is_ordered_for_k4_and_k5():
    a = np.arange(2 * 10 * 7).reshape(2, 10, 7)
    got = v2.split_action_tokens(a, 5)
    assert got.shape == (2, 5, 14)
    np.testing.assert_array_equal(got[0, 0], a[0, :2].reshape(-1))
    np.testing.assert_array_equal(got[0, -1], a[0, 8:].reshape(-1))
    got4 = v2.split_action_tokens(a, 4)
    assert got4.shape == (2, 4, 14)
    np.testing.assert_array_equal(got4[0, 0], a[0, [0, 1]].reshape(-1))
    np.testing.assert_array_equal(got4[0, -1], a[0, 8:].reshape(-1))
    with pytest.raises(ValueError):
        v2.split_action_tokens(a, 3)


def test_aligned_token_score_uses_matching_positions():
    q = torch.tensor([[[1., 0.], [0., 1.]]])
    m = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    got = v2.aligned_token_score(q, m)
    torch.testing.assert_close(got, torch.tensor([[1., 0.]]))


def test_maxsim_is_token_set_score():
    q = torch.tensor([[[1., 0.], [0., 1.]]])
    m = torch.tensor([[[0., 1.], [1., 0.]], [[1., 0.], [1., 0.]]])
    got = v2.maxsim_token_score(q, m)
    assert got.shape == (1, 2)
    torch.testing.assert_close(got, torch.tensor([[2., 1.]]))


def test_l0_prime_modes_are_finite_and_distinct():
    scores = torch.tensor([[2., 0., -1.], [0., 1., 2.]])
    errors = torch.tensor([[0., .02, .1], [.05, .01, 0.]])
    allowed = torch.ones_like(scores, dtype=torch.bool)
    dual = v2.l0_prime(scores, errors, allowed, mode="dual")
    single = v2.l0_prime(scores, errors, allowed, mode="single03")
    assert torch.isfinite(dual) and torch.isfinite(single)
    assert not torch.isclose(dual, single)


def test_l0_logits_are_not_temperature_scaled_and_gradient_matches_formula():
    scores = torch.tensor([[1.1, .7, -.3]], requires_grad=True)
    errors = torch.tensor([[0., .02, .09]])
    allowed = torch.tensor([[True, True, False]])
    loss = v2.l0_prime(scores, errors, allowed, mode="dual")
    loss.backward(); grad = scores.grad.detach().clone()
    s2 = scores.detach().clone().requires_grad_(True)
    mask = s2.masked_fill(~allowed, -1e4)
    lp = torch.log_softmax(mask, -1)
    t1 = torch.softmax((-errors / .01).masked_fill(~allowed, -1e4), -1)
    t5 = torch.softmax((-errors / .05).masked_fill(~allowed, -1e4), -1)
    expected = -(t1 * lp).sum(-1).mean() + (t5 * (t5.clamp_min(1e-12).log() - lp)).sum(-1).mean()
    expected.backward()
    torch.testing.assert_close(loss.detach(), expected.detach())
    torch.testing.assert_close(grad, s2.grad)
    # The single-temperature control is exactly the legacy target-only L0.
    torch.testing.assert_close(v2.l0_prime(scores.detach(), errors, allowed, mode="single03"),
                               v2.legacy_l0(scores.detach(), errors, allowed, tau=.03))


def test_r0_legacy_model_and_l0_match_step1_old_path():
    torch.manual_seed(31)
    b, m, dim = 3, 7, 32
    old = v2.step1._make_model("A", 1, dim)
    hist = torch.randn(b, 10, 7); state = torch.randn(b, 8)
    tokens = torch.randn(b, 6, dim); mask = torch.ones(b, 6, dtype=torch.bool)
    pool = torch.randn(m, 10, 7)
    q, _ = old(hist, state, tokens, mask)
    keys = torch.nn.functional.normalize(old.key(pool.flatten(1)), dim=-1)
    scores = torch.nn.functional.normalize(q, dim=-1) @ keys.T * old.log_scale.exp()
    err = torch.rand(b, m); allowed = torch.ones(b, m, dtype=torch.bool)
    target = torch.as_tensor(v2.step1._soft_teacher(err.numpy(), allowed.numpy(), .01))
    old_loss = -(target * torch.log_softmax(scores, -1)).sum(-1).mean()
    torch.testing.assert_close(v2.legacy_l0(scores, err, allowed, tau=.01), old_loss)


def test_pool_template_fallback_returns_exact_allowed_minimum_templates():
    rng=np.random.default_rng(4)
    pool=rng.normal(size=(12,10,7)).astype(np.float32)
    err=rng.random((5,12),dtype=np.float32)
    allowed=np.ones((5,12),bool);allowed[0,:6]=False
    got,idx=v2.pool_template_key_actions(err,allowed,pool)
    expected=np.argmin(np.where(allowed,err,np.inf),axis=1)
    np.testing.assert_array_equal(idx,expected)
    np.testing.assert_array_equal(got,pool[expected])
    np.testing.assert_allclose(got.mean((0,1)),pool[expected].mean((0,1)),atol=0,rtol=0)


def test_real_fit_cache_pool_template_roundtrip_at_least_50_candidates():
    path=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/scene-invariance-01/fit_rows.npz")
    if not path.exists():pytest.skip(f"real fit cache unavailable: {path}")
    z=np.load(path,allow_pickle=False)
    required={"pool_shape","error","allowed","fit"}
    if not required.issubset(z.files):pytest.skip(f"fit cache lacks {sorted(required-set(z.files))}")
    ids=np.flatnonzero(np.asarray(z["fit"],bool))[:64]
    if len(ids)<50:pytest.skip(f"fewer than 50 fit rows in cache: {len(ids)}")
    keys,pool_ids=v2.pool_template_key_actions(z["error"][ids],z["allowed"][ids],z["pool_shape"])
    expected=np.argmin(np.where(z["allowed"][ids],z["error"][ids],np.inf),axis=1)
    assert len(np.unique(pool_ids))>=50, f"only {len(np.unique(pool_ids))} unique nearest pool templates"
    np.testing.assert_array_equal(pool_ids,expected)
    np.testing.assert_array_equal(keys,z["pool_shape"][pool_ids])
    # Fallback has no action-to-template residual: every returned axis is the
    # saved template axis, so all per-axis statistics agree exactly.
    np.testing.assert_allclose(keys.mean((0,1)),z["pool_shape"][pool_ids].mean((0,1)),atol=1e-7,rtol=0)
    np.testing.assert_allclose(keys.std((0,1)),z["pool_shape"][pool_ids].std((0,1)),atol=1e-7,rtol=0)


def test_history_position_embeddings_distinguish_temporal_order():
    torch.manual_seed(12)
    net=v2._build_model("A",k=5,token_dim=16,d=32)
    h=torch.randn(2,10,7)
    swapped=h.flip(1)
    a=net.history_tokens(h);b=net.history_tokens(swapped)
    assert not torch.allclose(a,b)
    tok=torch.randn(2,12,16);state=torch.randn(2,8);groups=torch.tensor([1]*12)
    q1,_=net.encode_query(tok,h,state,token_groups=groups)
    q2,_=net.encode_query(tok,swapped,state,token_groups=groups)
    assert not torch.allclose(q1,q2)


def test_third_person_mask_is_training_only_and_true_means_ignore(monkeypatch):
    torch.manual_seed(7)
    net=v2._build_model("B",k=5,token_dim=16,d=32)
    tok=torch.randn(2,12,16);groups=torch.tensor([0]*4+[1]*4+[2]*4)
    hist=torch.randn(2,10,7);state=torch.randn(2,8)
    altered=tok.clone();altered[:, :4]+=100.
    monkeypatch.setattr(torch,"rand",lambda *args,**kwargs: torch.zeros(*args,device=kwargs.get("device","cpu")))
    net.train()
    a,_=net.encode_query(tok,hist,state,token_groups=groups,train_mask=True)
    b,_=net.encode_query(altered,hist,state,token_groups=groups,train_mask=True)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    net.eval()
    c,_=net.encode_query(tok,hist,state,token_groups=groups,train_mask=True)
    d,_=net.encode_query(altered,hist,state,token_groups=groups,train_mask=True)
    assert not torch.allclose(c,d)


def test_scene_stratified_batches_use_unique_scenes_and_multiple_scenes():
    scene = np.repeat(np.array(["a", "b", "c", "d"]), 40)
    ids = np.arange(len(scene))
    batches = list(v2.scene_stratified_indices(scene, ids, batch_size=32, scenes_per_batch=4, per_scene=8))
    assert batches
    for batch in batches:
        assert len(batch) == 32
        names = scene[batch]
        assert len(set(names.tolist())) == 4
        for name in set(names.tolist()):
            assert (names == name).sum() == 8


def test_scene_stratified_batch_forces_same_scene_different_type_pair():
    scene=np.repeat(np.array(["a","b","c","d"]),40);ids=np.arange(len(scene))
    pair={"c":(80,81)}
    batch=next(v2.scene_stratified_indices(scene,ids,batch_size=32,scenes_per_batch=4,
        per_scene=8,seed=2,different_type_pairs=pair))
    assert 80 in batch and 81 in batch
    assert scene[80] == scene[81] == "c"


def test_scene_stratified_sampler_excludes_held_out_group_rows():
    groups=np.repeat(np.array(["fit_a","fit_b","held"]),40)
    ids=np.arange(len(groups))
    batch=next(v2.scene_stratified_indices(groups,ids,batch_size=16,scenes_per_batch=2,
        per_scene=8,seed=5,excluded_scenes=("held",)))
    assert len(batch)==16 and not np.any(groups[batch]=="held")


def test_relational_loss_ignores_same_scene_near_pairs_and_adds_cross_scene_positives():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=-1)
    key = torch.nn.functional.normalize(torch.randn(4, 5, 8), dim=-1)
    # Pair 0/1 is identical and shares scene: ignored. Pair 0/2 is near and
    # cross-scene: positive. Pair 0/3 is deliberately distant: negative.
    action = torch.randn(4, 10, 6)
    action[1] = action[0]
    action[2] = action[0] + .001
    action[3] = -action[0]
    _,positive,valid=v2.relational_pair_masks(action,np.array(["s0","s0","s1","s2"]),.01)
    assert not positive[0,1] and not valid[0,1]  # near same-scene pair is ignored
    assert valid[0,3] and not positive[0,3]      # same/different-type negatives stay valid
    assert positive[0,2] and valid[0,2]
    loss, positives = v2.relational_soft_infonce(
        q, key, action, np.array(["s0", "s0", "s1", "s2"]), tau_pos=.01)
    assert torch.isfinite(loss)
    assert positives == int(positive.sum())


def test_model_returns_k_separate_normalized_query_tokens_for_all_modes():
    torch.manual_seed(1)
    b, n, dim = 3, 24, 32
    token = torch.randn(b, n, dim)
    hist = torch.randn(b, 10, 7)
    state = torch.randn(b, 8)
    groups = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    for k in (4, 5):
        for kind, pooled in [("A", False), ("B", False), ("B", True)]:
            model = v2._build_model(kind, k=k, token_dim=dim, pooled=pooled)
            q, weights = model.encode_query(token, hist, state, token_groups=groups, train_mask=True)
            assert q.shape == (b, k, 256)
            torch.testing.assert_close(q.norm(dim=-1), torch.ones(b, k), atol=1e-5, rtol=1e-5)
            assert weights["vlm"].shape[-1] in (8, 24)
            keys = model.encode_keys(torch.randn(6, 10, 7))
            assert keys.shape == (6, k, 256)
            assert model.scores(q, keys).shape == (b, 6)


def test_selection_grid_has_at_most_ten_fits_and_seed_is_singleton():
    grid = v2.config_grid()
    assert len(grid) == 4
    assert {x["kind"] for x in grid} == {"A", "B"}
    assert {x["k"] for x in grid} == {4, 5}
    assert set(v2.CONDITIONS) == {"V2-A", "V2-A-L1", "V2-B", "V2-B-L1", "V2-B-pool"}
    assert v2.SEED == 4101


def test_ablation_stage_switches_are_single_cumulative_config_steps():
    r0=v2.ablation_config("R0");r1=v2.ablation_config("R1");r2=v2.ablation_config("R2")
    r3=v2.ablation_config("R3");r4=v2.ablation_config("R4");r5=v2.ablation_config("R5",l1_weight=.3)
    assert r0["implementation"]=="step1_legacy" and r0["lr"]==3e-4
    assert r1["k"]==1 and r1["key_mode"]=="legacy" and r1["query_pool"] and r1["no_history_cross"]
    assert r1["l0"]=="legacy" and not r1["l1"]
    assert r2["kind"]=="B" and r2["k"]==1 and r2["key_mode"]=="legacy"
    assert r3["l1"] and r3["l0"]=="legacy" and r4["k"]==5 and r4["key_mode"]=="split"
    assert r5["l0"]=="dual" and r5["l1"] and r5["l1_weight"]==.3
    with pytest.raises(ValueError):v2.ablation_config("R1",k=5)


def test_r1_has_single_z0_old_key_old_loss_and_skips_history_cross():
    torch.manual_seed(18)
    cfg=v2.ablation_config("R1")
    model=v2._build_model(cfg["kind"],k=cfg["k"],token_dim=16,d=32,
        key_mode=cfg["key_mode"],query_pool=cfg["query_pool"],
        no_history_cross=cfg["no_history_cross"])
    assert model.z0.shape==(1,1,32)
    called=[];handle=model.ctx_cross.register_forward_hook(lambda *_:called.append(True))
    q,_=model.encode_query(torch.randn(2,8,16),torch.randn(2,10,7),torch.randn(2,8),
                           token_groups=torch.ones(8,dtype=torch.long))
    handle.remove()
    assert q.shape==(2,1,32) and not called
    assert model.key_mode=="legacy" and cfg["l0"]=="legacy" and not cfg["l1"]


def test_stage_acceptance_rule_and_next_stage_inherit_retained_configuration():
    incumbent={"top1_type_error":.02,"spearman_positive":.4}
    accepted=v2.stage_acceptance({"top1_type_error":.0203,"spearman_positive":.41},incumbent)
    assert accepted["accepted"] and accepted["error_ok"] and accepted["spearman_ok"]
    rejected_error=v2.stage_acceptance({"top1_type_error":.0205,"spearman_positive":.8},incumbent)
    rejected_rho=v2.stage_acceptance({"top1_type_error":.019,"spearman_positive":.4},incumbent)
    assert not rejected_error["accepted"] and not rejected_error["error_ok"]
    assert not rejected_rho["accepted"] and not rejected_rho["spearman_ok"]
    retained=v2.ablation_config("R1")
    r2=v2.stage_candidate_configs(retained,"R2")
    assert r2["kind"]=="B" and r2["k"]==1 and r2["query_pool"]
    candidates=[
        {"config":v2.stage_candidate_configs(r2,"R3"),"top1_type_error":.0204,"spearman_positive":.42},
        {"config":dict(v2.stage_candidate_configs(r2,"R3"),l1_weight=.3),"top1_type_error":.0201,"spearman_positive":.44},
    ]
    chosen,checked=v2.choose_stage_candidate(candidates,incumbent,"R3")
    assert chosen is checked[1] and chosen["config"]["l1_weight"]==.3
    # If the only proposed patch fails, caller retains the prior model; the
    # following stage is composed from that unchanged configuration.
    rejected,checked=v2.choose_stage_candidate([candidates[0]],
        {"top1_type_error":.019,"spearman_positive":.5},"R3")
    assert rejected is None and not checked[0]["accepted"]
    next_candidate=v2.stage_candidate_configs(r2,"R4")
    assert next_candidate["kind"]==r2["kind"] and next_candidate["l1"]==r2["l1"]
    assert next_candidate["k"]==5 and next_candidate["key_mode"]=="split"


def test_l1_action_key_loader_returns_pool_templates(tmp_path):
    rng=np.random.default_rng(27)
    pool=rng.normal(size=(9,10,7)).astype(np.float32)
    error=rng.random((4,9),dtype=np.float32);allowed=np.ones((4,9),bool);allowed[1,:4]=False
    data={"row_id":np.arange(4),"pool_shape":pool,"error":error,"allowed":allowed}
    actions=v2.load_training_actions(data,tmp_path)
    expected=np.argmin(np.where(allowed,error,np.inf),axis=1)
    np.testing.assert_array_equal(actions,pool[expected])
    np.testing.assert_array_equal(np.load(tmp_path/"training_pool_template_keys_v2.npy"),pool[expected])


def test_eval_refuses_to_use_step1_metrics_before_api_handoff(tmp_path):
    # Avoid reading any Eval labels or silently substituting a private evaluator.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        v2.evaluate(tmp_path, source_out=tmp_path, device="cpu")


def test_multiseed_gate_uses_r0_mean_and_twice_its_sample_sd():
    baseline={"top1_type_error_mean":.02,"spearman_positive_mean":.40,"spearman_positive_sd":.01}
    good={"top1_type_error_mean":.0203,"spearman_positive_mean":.421}
    bad_error={"top1_type_error_mean":.0205,"spearman_positive_mean":.50}
    bad_rho={"top1_type_error_mean":.019,"spearman_positive_mean":.42}
    assert v2.multiseed_acceptance(good,baseline)["accepted"]
    assert not v2.multiseed_acceptance(bad_error,baseline)["accepted"]
    assert not v2.multiseed_acceptance(bad_rho,baseline)["accepted"]


def test_seed_aggregation_reports_sample_standard_deviation():
    rows=[{"seed":4101,"top1_type_error":1.,"recovery":.2,"spearman_positive":.3,"mean_query_token_cosine":None},
          {"seed":4102,"top1_type_error":3.,"recovery":.4,"spearman_positive":.5,"mean_query_token_cosine":None},
          {"seed":4103,"top1_type_error":2.,"recovery":.6,"spearman_positive":.7,"mean_query_token_cosine":None}]
    got=v2.aggregate_seed_metrics(rows)
    assert got["top1_type_error_mean"]==2.
    assert got["top1_type_error_sd"]==1.
    assert got["recovery_mean"]==pytest.approx(.4)


def test_other_history_is_deterministic_and_from_a_different_episode():
    data={"history":np.arange(6*10*7,dtype=np.float32).reshape(6,10,7),
          "episode":np.array([10,10,20,20,30,30])}
    ids=np.array([0,2,4])
    got=v2.history_variant(data,ids,"other_episode",seed=9)
    again=v2.history_variant(data,ids,"other_episode",seed=9)
    np.testing.assert_array_equal(got["history"],again["history"])
    for row in ids:
        source=np.flatnonzero(data["episode"]!=data["episode"][row])
        assert any(np.array_equal(got["history"][row],data["history"][j]) for j in source)
    zero=v2.history_variant(data,ids,"zero")
    assert not zero["history"][ids].any()
    np.testing.assert_array_equal(zero["history"][1],data["history"][1])


def test_legacy_r0_logits_matches_untouched_old_forward_path():
    torch.manual_seed(84)
    model=v2.step1._make_model("A",1,32)
    hist=torch.randn(4,10,7);state=torch.randn(4,8);tokens=torch.randn(4,11,32)
    pool=torch.randn(13,10,7)
    mask=torch.ones(tokens.shape[:2],dtype=torch.bool)
    q,_=model(hist,state,tokens,mask)
    keys=torch.nn.functional.normalize(model.key(pool.flatten(1)),dim=-1)
    expected=torch.nn.functional.normalize(q,dim=-1)@keys.T*model.log_scale.exp()
    actual=v2.legacy_r0_logits(model,hist,state,tokens,pool)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    tokens_half=tokens.half().float();q_half,_=model(hist,state,tokens_half,mask)
    expected_half=torch.nn.functional.normalize(q_half,dim=-1)@keys.T*model.log_scale.exp()
    half=v2.legacy_r0_logits(model,hist,state,tokens.half(),pool)
    torch.testing.assert_close(half,expected_half,atol=0,rtol=0)
