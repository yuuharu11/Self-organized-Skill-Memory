"""Query variants: scale-only shapes (S1), pi0-FAST (F1), cross attention (X1)."""
import inspect

import numpy as np
import pytest

from tests.support import (
    UNSEEN,
    add_path,
    load_script,
)

so = load_script(UNSEEN / "scale_only_query.py")
fq = load_script(UNSEEN / "fast_query.py")
add_path(UNSEEN)
import l90x_rows as lr  # noqa: E402
import x_attn_query as xq  # noqa: E402

# ── Scale-only query ──────────────────────────────────────────────────────────────────────────────

def _rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _rotate(x, r):
    return np.concatenate([x[..., :3] @ r, x[..., 3:] @ r], -1)


def test_scale_only_error_has_no_rotation():
    rng = np.random.default_rng(0); u = rng.normal(size=(1, 10, 6))
    q = _rotate(u[0], _rot(np.pi / 2))[None] * 2.0
    for cond in ('S-shared', 'S-sep'):
        assert so.scale_only_error(q, u, cond)[0, 0] > .1          # a rotated copy is NOT a match
        assert so.scale_only_error(u * 3.0, u, cond)[0, 0] < 1e-12  # a scaled copy is a perfect match
    # R-so3 control does remove the rotation
    e = so.procrustes_error_w(q.astype(np.float32), u.astype(np.float32), 'cpu')
    assert e[0, 0] < 1e-8


def test_scale_only_error_matches_bruteforce_and_clamps_negative_scale():
    rng = np.random.default_rng(1); u = rng.normal(size=(4, 10, 6)); q = rng.normal(size=(3, 10, 6))
    grid = np.linspace(0, 5, 20001)
    for cond in ('S-shared', 'S-sep'):
        E = so.scale_only_error(q, u, cond)
        for i in range(3):
            for k in range(4):
                tot = 0.
                for g in so.groups_of(cond):
                    d = ((grid[:, None, None] * u[k][None, :, g] - q[i][None, :, g]) ** 2).sum((1, 2))
                    tot += d.min()
                assert E[i, k] == pytest.approx(tot / 60, rel=1e-4, abs=1e-7)
    # anti-parallel: best non-negative scale is 0 -> error = ||q||^2
    E = so.scale_only_error(-u[:1], u[:1], 'S-shared')
    assert E[0, 0] == pytest.approx((u[0] ** 2).mean(), rel=1e-6)


def test_closed_form_scale_correct_and_separate():
    rng = np.random.default_rng(2); u = rng.normal(size=(5, 10, 6))
    q = u.copy(); q[..., :3] *= 2.0; q[..., 3:] *= .5
    s = so.closed_form_scale(u, q, 'S-sep'); np.testing.assert_allclose(s, np.tile([2., .5], (5, 1)), rtol=1e-10)
    s1 = so.closed_form_scale(u, u * 1.7, 'S-shared'); np.testing.assert_allclose(s1[:, 0], 1.7, rtol=1e-10)
    np.testing.assert_allclose(so.apply_scales(u, s, 'S-sep'), q, rtol=1e-10)


def test_match_speed_keeps_direction_and_sets_rms():
    rng = np.random.default_rng(3); u = rng.normal(size=(2, 10, 6)); ref = rng.normal(size=(2, 10, 6)) * 3
    out, _ = so.match_speed(u, ref, 'S-sep')
    for g in so.groups_of('S-sep'):
        np.testing.assert_allclose(so.rms_speed(out, g), so.rms_speed(ref, g))
        cos = (out[..., g] * u[..., g]).sum((1, 2)) / np.linalg.norm(out[..., g].reshape(2, -1), axis=1) / np.linalg.norm(u[..., g].reshape(2, -1), axis=1)
        np.testing.assert_allclose(cos, 1.)


def test_type_of_is_scale_invariant_but_direction_sensitive():
    rng = np.random.default_rng(4); x = rng.normal(size=(10, 6))
    np.testing.assert_allclose(so.type_of(x * 5, 'S-shared'), so.type_of(x, 'S-shared'))
    assert not np.allclose(so.type_of(_rotate(x, _rot(1.)), 'S-shared'), so.type_of(x, 'S-shared'))


def test_dct3_roundtrip_of_low_frequency_motion():
    from scipy.fft import idct
    c = np.zeros((12, 6)); c[:3] = np.random.default_rng(5).normal(size=(3, 6)); x = idct(c, type=2, norm='ortho', axis=0)
    np.testing.assert_allclose(so.dct3_decode(so.dct3_values(x), 12), x, atol=1e-10)


def _w_inputs(n=4):
    rng = np.random.default_rng(6)
    return {'hist': rng.uniform(-1, 1, (n, 10, 7)).astype(np.float32), 'state': rng.normal(size=(n, 8)).astype(np.float32),
            'has_hist': np.array([True, False, True, True][:n]), 'pi05': rng.uniform(-1, 1, (n, 10, 7)).astype(np.float32),
            'frame': np.array([10, 0, 20, 30][:n])}


def test_generation_refuses_ground_truth_and_ignores_it():
    rng = np.random.default_rng(7); pool = {'phys': rng.normal(size=(6, 10, 6)) * .1}
    inp = _w_inputs(); pick = np.array([0, 1, 2, 3])
    for key in ('target', 'actions', 'y'):
        with pytest.raises(ValueError):
            so.generate_w(pool, pick, 'S-shared', 'hist', {**inp, key: np.zeros((4, 10, 7))})
    # generation signatures carry no target argument
    for fn in (so.generate_w, so.generate_u):
        assert not any(p in inspect.signature(fn).parameters for p in so.TARGET_KEYS)
    # query inputs refuse targets too
    with pytest.raises(ValueError):
        so.query_inputs({**inp, 'target': np.zeros(1)}, 'P5')
    a = so.generate_w(pool, pick, 'S-sep', 'pi05', inp); b = so.generate_w(pool, pick, 'S-sep', 'pi05', dict(inp))
    np.testing.assert_array_equal(a, b)


def test_hist_rule_falls_back_to_stored_without_history():
    rng = np.random.default_rng(8); pool = {'phys': rng.normal(size=(6, 10, 6)) * .1}
    inp = _w_inputs(); pick = np.array([0, 1, 2, 3])
    st = so.generate_w(pool, pick, 'S-shared', 'stored', inp); hi = so.generate_w(pool, pick, 'S-shared', 'hist', inp)
    np.testing.assert_allclose(st[1], hi[1]); assert not np.allclose(st[0], hi[0])


def test_u_generation_uses_stored_length_not_target_length():
    rng = np.random.default_rng(9)
    pool = {'values': rng.normal(size=(3, 18)) * .05, 'lengths': np.array([5, 9, 14])}
    inp = {'n_past': np.array([0, 4, 10]), 'past': rng.uniform(-1, 1, (3, 10, 7)), 'n_pi05': np.array([3, 10, 1]),
           'pi05_now': rng.uniform(-1, 1, (3, 10, 7))}
    for cond in so.CONDS:
        for rule in ('stored', 'hist', 'pi05'):
            out = so.generate_u(pool, np.array([2, 0, 1]), cond, rule, inp)
            assert [len(o) for o in out] == [14, 5, 9]


def test_u_errors_full_and_overlap_frames():
    p = [np.zeros((3, 6)), np.ones((6, 6))]; y = [np.ones((5, 7)), np.zeros((4, 7))]
    e = so.u_errors(p, y, np.zeros(2))
    assert e['n_overlap'].tolist() == [3, 4]
    assert e['full'][0] == pytest.approx(6 / 7) and e['overlap'][1] == pytest.approx(6 / 7)


def test_exclusions_own_task_and_holdout():
    pool_task = np.array([1, 2, 3, 2]); keep = np.array([True, True, False, True])
    A = so.allowed_matrix(np.array([2, 5]), pool_task, keep, True)
    assert A.tolist() == [[True, False, False, False], [True, True, False, True]]
    A2 = so.allowed_matrix(np.array([2]), pool_task, keep, False)   # unseen queries: holdout units only removed
    assert A2.tolist() == [[True, True, False, True]]
    with pytest.raises(ValueError):
        so.allowed_matrix(np.array([1]), np.array([1]), np.array([True]), True)


def test_holdout_tasks_and_selection_never_mention_eval():
    assert so.HOLDOUT == (2, 7, 12, 17, 22, 27, 32, 37)
    src = inspect.getsource(so.train_one) + inspect.getsource(so.select) + inspect.getsource(so.load_train_data)
    assert "'eval'" not in src and 'UNSEEN' not in src
    # Eval actions are only read inside evaluate (scoring)
    assert "lu('actions')[eval_ids" not in inspect.getsource(so.prepare)
    assert 'eval_targets_read' in inspect.getsource(so.prepare)


def test_teacher_tau_relative_to_gap():
    E = np.array([[0., 1., 2.], [1., 1., 4.]]); A = np.ones_like(E, bool)
    # gaps: mean-min = 1 and 1 -> median 1
    assert so.teacher_tau(E, A, .1) == pytest.approx(.1)


# ── pi0-FAST query ────────────────────────────────────────────────────────────────────────────────

def _sp_or_skip():
    try:
        return fq._paligemma_sp()
    except Exception as e:  # pragma: no cover - needs the cached PaliGemma tokenizer
        pytest.skip(f'PaliGemma tokenizer unavailable: {e}')


def _decoder_or_skip():
    if not (fq.FAST_TOKENIZER / 'tokenizer.json').exists():
        pytest.skip('FAST tokenizer files not present')
    return fq.FastActionDecoder(paligemma=_sp_or_skip())


def _ns():
    st = {'mean': np.arange(8) * .1, 'std': np.full(8, .5), 'q01': np.full(8, -2.), 'q99': np.full(8, 2.)}
    ac = {'mean': np.r_[np.full(6, -.3), 0.], 'std': np.full(7, .4), 'q01': np.full(7, -1.), 'q99': np.full(7, 1.)}
    return {'state': st, 'actions': ac}


def test_norm_roundtrips():
    ns = _ns(); x = np.random.default_rng(0).normal(size=(5, 10, 7))
    np.testing.assert_allclose(fq.zunnorm(fq.znorm(x, ns['actions']), ns['actions']), x, atol=1e-12)
    np.testing.assert_allclose(fq.qunnorm(fq.qnorm(x, ns['actions']), ns['actions']), x, atol=1e-12)


def test_fast_to_common_adds_state_on_six_axes_only():
    fast_ns, pi05_ns = _ns(), _ns()
    rng = np.random.default_rng(1); raw = rng.normal(size=(3, 10, 7)) * .3; state = rng.normal(size=(3, 8))
    delta = raw.copy(); delta[..., :6] -= state[:, None, :6]                   # training-time DeltaActions
    model_space = fq.znorm(delta, fast_ns['actions'])
    common, back = fq.fast_to_common(model_space, state, fast_ns, pi05_ns)
    np.testing.assert_allclose(back, raw, atol=1e-5)                          # gripper (axis 6) not shifted
    np.testing.assert_allclose(common, fq.qnorm(raw, pi05_ns['actions']), atol=1e-5)


def test_fast_prefix_format_and_masks():
    sp = _sp_or_skip(); st = np.array([-1., -.5, 0., .5, .999, 1.5, -2., .1])
    tok, mask, ar, loss = fq.fast_prefix(sp, 'Open_the Drawer ', st)
    n = int(mask.sum()); assert tok.shape == (fq.MAX_TOKEN_LEN,) and not ar.any() and not loss.any()
    text = sp.decode(tok[:n].tolist())
    bins = np.digitize(st, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    assert text == f"Task: open the drawer, State: {' '.join(map(str, bins))};\n"
    assert (tok[n:] == 0).all()


def test_fast_decoder_roundtrip_and_extract():
    dec = _decoder_or_skip(); sp = dec.pg
    t = np.linspace(0, 1, 10)[:, None]
    chunk = np.concatenate([np.sin(np.pi * t + np.arange(6) * .3) * .8, np.ones((10, 1))], 1)
    ids = dec.encode(chunk)
    coeff, back = dec.decode_fast(ids)
    assert coeff.shape == (10, 7)
    assert np.abs(back - chunk).max() < .1                                     # quantisation of DCT at scale 10
    pg = (sp.vocab_size() - 1 - fq.FAST_SKIP - np.asarray(ids)).tolist()
    out = sp.encode('Action: ') + pg + sp.encode('|') + [fq.PALIGEMMA_EOS] + [5, 6, 7] + [0] * 20
    r = dec.extract(np.asarray(out))
    assert r['ok'] and r['eos'] and r['n_generated'] == out.index(fq.PALIGEMMA_EOS)
    np.testing.assert_allclose(r['actions'], back); np.testing.assert_array_equal(r['coeff'], coeff)
    bad = dec.extract(np.asarray(sp.encode('nothing here') + [fq.PALIGEMMA_EOS]))
    assert not bad['ok'] and (bad['actions'] == 0).all()


def test_nearest_scale_only_keeps_direction_and_respects_mask():
    rng = np.random.default_rng(2); pool = rng.normal(size=(6, 10, 6))
    allowed = np.ones((3, 6), bool); allowed[2, 4] = False
    q = np.stack([pool[3] * 2.5, -pool[1] * 1.0 + 0 * pool[0], pool[4] * .7])
    pick, d = fq.nn_scale_only(q, pool, allowed)
    assert pick[0] == 3 and pick[1] != 1 and pick[2] != 4 and np.isinf(d[2, 4])


def test_nn_dct_is_scale_invariant():
    so = fq._so(); rng = np.random.default_rng(3)
    vals = rng.normal(size=(5, 18)); phys16 = np.stack([so.dct3_decode(v, so.L16) for v in vals])
    pool_type = fq.dct_type(vals, phys16); allowed = np.ones((2, 5), bool)
    pick, _ = fq.nn_dct(np.stack([vals[2] * 3., vals[4] * .2]), pool_type, allowed)
    assert list(pick) == [2, 4]


def test_rerank_limits():
    rng = np.random.default_rng(4); sc = rng.normal(size=(4, 9)); dist = rng.random((4, 9)); A = np.ones((4, 9), bool); A[:, 0] = False
    np.testing.assert_array_equal(fq.rerank(sc, dist, A, 1), np.where(A, sc, -np.inf).argmax(1))
    np.testing.assert_array_equal(fq.rerank(sc, dist, A, 8), np.where(A, dist, np.inf).argmin(1))


def test_spearman_rows_and_mask():
    a = np.array([[1., 2., 3., 4., 100.], [4., 3., 2., 1., -100.]]); b = np.array([[10., 20., 30., 40., -5.], [10., 20., 30., 40., 5.]])
    A = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], bool)
    sp, pe = fq.spearman_rows(a, b, A)
    np.testing.assert_allclose(sp, [1., -1.]); np.testing.assert_allclose(pe, [1., -1.])


def test_cosine_rows_nan_for_zero_vectors():
    c = fq.cosine_rows(np.array([[1., 0, 0], [0, 0, 0], [0, 1, 0]]), np.array([[2., 0, 0], [1, 0, 0], [0, -1, 0]]))
    assert c[0] == pytest.approx(1.) and np.isnan(c[1]) and c[2] == pytest.approx(-1.)


def test_rot180_view():
    img = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3); rows = {'image_base': img, 'image_wrist': img}
    a, _ = fq.images_of(rows, np.array([0, 1]), 'asis'); b, _ = fq.images_of(rows, np.array([0, 1]), 'rot180')
    np.testing.assert_array_equal(a, img); np.testing.assert_array_equal(b, img[:, ::-1, ::-1])


def test_ground_truth_only_in_evaluate():
    for fn in (fq.infer_fast, fq.infer_pi05, fq.source_rows, fq.fast_obs, fq.nn_scale_only, fq.nn_dct, fq.u_query, fq.p5_scores, fq.rerank):
        src = inspect.getsource(fn)
        assert 'actions.npy' not in src and "'actions')" not in src
    ev = inspect.getsource(fq.evaluate)
    assert "UNSEEN / 'actions.npy'" in ev and '# scoring only' in ev


def test_k_is_selected_on_dev_only():
    src = inspect.getsource(fq.aggregate)
    sel = src[src.index('choose k'):src.index('for split in')]
    assert "|dev||" in sel and 'eval' not in sel


# ── Cross-attention query ─────────────────────────────────────────────────────────────────────────

def test_l90x_task_split_excludes_fixed_dev_eval_and_disjoint():
    tasks,fit,select=lr.choose_tasks(smoke=True)
    ids={int(t['task_id']) for t in tasks}
    assert len(tasks)==4 and len(fit)==len(select)==2
    assert not ids.intersection(lr.EVAL_TASKS+lr.DEV_TASKS)
    assert not set(fit).intersection(select)

def test_padded_l90x_history_uses_active_action_channels():
    padded=np.zeros((10,32),np.float32)
    padded[:,:7]=np.arange(70,dtype=np.float32).reshape(10,7)
    np.testing.assert_array_equal(lr.active_action_channels(padded),padded[:,:7])
    assert lr.active_action_channels(padded).any()
    with pytest.raises(ValueError,match='at least seven'):
        lr.active_action_channels(np.zeros((10,6),np.float32))

def test_history_corruption_contract():
    h=np.arange(4*10*7,dtype=np.float32).reshape(4,10,7)
    zero=xq.corrupt_history(h,'zero');rev=xq.corrupt_history(h,'reverse')
    assert np.count_nonzero(zero)==0
    assert np.array_equal(rev[:,:,:3],-h[:,:,:3])
    assert np.array_equal(rev[:,:,3:],h[:,:,3:])
    task=np.array([0,0,1,1]);sw=xq.corrupt_history(h,'swap',seed=3,task=task)
    assert all(any(np.array_equal(sw[i],h[j]) for j in range(4) if task[j]!=task[i]) for i in range(4))

def test_requested_corruption_mix_is_exact_and_phase_shift_is_same_episode():
    codes=xq.corruption_codes(100000,seed=4101)
    freq={k:float(np.mean(codes==k)) for k in xq.CORRUPT}
    assert all(abs(freq[k]-xq.CORRUPT[k])<.015 for k in freq)
    selected=xq.corruption_codes(100000,seed=4102,selection=True)
    sfreq={k:float(np.mean(selected==k)) for k in xq.SELECTION_CORRUPT}
    assert all(abs(sfreq[k]-xq.SELECTION_CORRUPT[k])<.015 for k in sfreq)
    assert abs(sfreq['true']-.5)<.015
    ds={'hist':np.arange(4*10*7,dtype=np.float32).reshape(4,10,7),
        'task':np.array([1,1,1,2]),'episode':np.array([4,4,4,4]),'frame':np.array([0,10,20,0])}
    shifted,valid=xq.phase_shift_history(ds,10)
    assert valid.tolist()==[0,1]
    np.testing.assert_array_equal(shifted[0],ds['hist'][1]);np.testing.assert_array_equal(shifted[1],ds['hist'][2])

def test_training_mixture_applies_same_episode_phase_shift_histories():
    n=1000
    ds={'hist':np.arange(n*10*7,dtype=np.float32).reshape(n,10,7),
        'task':np.ones(n,np.int64),'episode':np.full(n,5,np.int64),'frame':np.arange(n,dtype=np.int64)*10}
    ids=np.arange(n)
    mixed,modes=xq._mixed_history(ds,ids,4103)
    phase=np.flatnonzero(modes=='phase')
    assert len(phase)>150
    lookup={(int(t),int(e),int(f)):i for i,(t,e,f) in enumerate(zip(ds['task'],ds['episode'],ds['frame']))}
    for i in phase[:20]:
        source=next(lookup[(1,5,int(ds['frame'][i])+s)] for s in (-20,-10,10,20)
                    if (1,5,int(ds['frame'][i])+s) in lookup and
                    np.array_equal(mixed[i],ds['hist'][lookup[(1,5,int(ds['frame'][i])+s)]]))
        assert source != i

def test_conditional_metrics_topk_recall_ndcg_and_gap():
    err=np.array([[0.,.01,.02,.03,.04,.05,.06,.07,.08,.09]],np.float32)
    score=-err.copy();c=xq._conditional_spearman_rows(score,err);m=xq._retrieval_row_metrics(score,err)
    assert c[0]>.99 and m['recall@1'][0] and m['recall@5'][0] and m['ndcg@10'][0]>.99
    assert m['recall@20'][0]
    assert m['gap@5'][0]>.99

def test_eval_targets_rejected_from_fit_select():
    with pytest.raises(ValueError):xq.assert_eval_boundary({'split':'eval'})
    with pytest.raises(ValueError):xq.assert_eval_boundary({'eval_actions':np.zeros((1,10,7))})
    fit,sel=xq.split_fit_select([False,False,True,True])
    assert np.array_equal(fit,[0,1]) and np.array_equal(sel,[2,3])
    with pytest.raises(ValueError):xq.split_fit_select([False,False])

@pytest.mark.parametrize('arch',['P5','XH','XC','XA'])
def test_model_scores_are_real_per_candidate(arch):
    torch=pytest.importorskip('torch');torch.set_num_threads(8);torch.manual_seed(1)
    model=xq.make_model(arch,78,60,64,expert_dim=32)
    feat={'hist':torch.randn(2,10,7),'state':torch.randn(2,8),'visual':torch.randn(2,32,64)}
    if arch=='XA':feat.update(pi05=torch.randn(2,10,7),expert=torch.randn(2,10,32))
    scores=model(feat,torch.randn(9,60))
    assert scores.shape==(2,9) and torch.isfinite(scores).all()

def test_training_loss_ignores_disallowed_candidates_in_both_terms():
    torch=pytest.importorskip('torch');torch.set_num_threads(8)
    allowed=torch.tensor([[True,True,False]])
    teacher=torch.tensor([[.75,.25,0.]])
    errors=torch.tensor([[0.,1.,100.]])
    a=torch.tensor([[1.,0.,-100.]])
    b=torch.tensor([[1.,0.,100.]])
    loss_a=xq.masked_training_loss(a,teacher,errors,allowed)
    loss_b=xq.masked_training_loss(b,teacher,errors,allowed)
    torch.testing.assert_close(loss_a,loss_b)
    with pytest.raises(ValueError,match='disallowed'):
        xq.masked_training_loss(a,torch.tensor([[.5,.25,.25]]),errors,allowed)

def test_top3_layer_readout_concatenates_final_v16_v17():
    import torch;torch.set_num_threads(8)
    ds={'hist':np.zeros((2,10,7),np.float32),'state':np.zeros((2,8),np.float32),'phi':np.array([.2,.7],np.float32),
        'image':np.zeros((2,128,2048),np.float32),'mid16':np.ones((2,128,256),np.float32),
        'mid17':np.full((2,128,256),2,np.float32)}
    feat=xq._feature(ds,np.array([0,1]),'top3',arch='XH',device='cpu')
    assert tuple(feat['visual'].shape)==(2,32,2560)
    model=xq.make_model('XH',78,60,2560)
    assert model(feat,torch.randn(5,60)).shape==(2,5)

def test_temporal_image_delta_is_exact_previous_window_and_has_flag():
    torch=pytest.importorskip('torch');torch.set_num_threads(8)
    task=np.array([1,1,1,1]);epi=np.array([2,2,2,3]);frame=np.array([0,10,20,10])
    np.testing.assert_array_equal(xq.predecessor_indices(task,epi,frame),[-1,0,1,-1])
    image=np.stack([np.full((128,2),v,np.float32) for v in (2,5,9,20)])
    ds={'task':task,'episode':epi,'frame':frame,'prev_index':xq.predecessor_indices(task,epi,frame),
        'hist':np.ones((4,10,7),np.float32),'state':np.zeros((4,8),np.float32),'image':image,
        'phi':np.array([0,.2,.4,0],np.float32)}
    feat=xq._feature(ds,np.array([1,2,3]),'final',arch='P5-imgdelta',device='cpu')
    np.testing.assert_allclose(feat['visual_delta'].numpy()[:,0,0],[3,4,0])
    np.testing.assert_allclose(feat['context'].numpy()[:,3],[1,1,0])
    assert feat['context'].shape==(3,4)

def test_temporal_context_model_does_not_use_episode_phase_as_delta():
    torch=pytest.importorskip('torch');torch.set_num_threads(8);torch.manual_seed(2)
    model=xq.make_model('P5-imgdelta',78,60,64).eval()
    feat={'hist':torch.randn(2,10,7),'state':torch.randn(2,8),'visual':torch.randn(2,32,64),
          'visual_delta':torch.randn(2,32,64),'context':torch.randn(2,4),
          'query_phi':torch.tensor([.1,.7])}
    cand=torch.randn(9,60)
    a=model(feat,cand);feat['query_phi']=torch.tensor([.9,.01]);b=model(feat,cand)
    torch.testing.assert_close(a,b)
    assert a.shape==(2,9) and torch.isfinite(a).all()

def test_query_conditioned_fusion_gate_outputs_row_specific_weight():
    torch=pytest.importorskip('torch');torch.set_num_threads(8);torch.manual_seed(3)
    gate=xq.make_fusion_gate(14)
    with torch.no_grad():
        gate[0].weight.fill_(.03);gate[0].bias.zero_();gate[2].weight.fill_(.02);gate[2].bias.zero_()
    p5=torch.tensor([[3.,1.,0.],[.1,.0,-.1]])
    nn=torch.tensor([[0.,2.,1.],[3.,0.,-1.]])
    feat={'hist':torch.ones(2,10,7),'state':torch.ones(2,8),'context':torch.tensor([[0.,0.,0.,0.],[2.,1.,3.,1.]])}
    scores,alpha=xq.combine_with_gate(gate,p5,nn,feat)
    assert scores.shape==(2,3) and alpha.shape==(2,)
    assert torch.all((alpha>0)&(alpha<1)) and not torch.isclose(alpha[0],alpha[1])
