import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[2] / "skill_moe/skillnet/examples/memory_pilot/unseen_libero"
sys.path.insert(0, str(HERE))
import key_latent_sources as src


def test_fixed_ax_noise_is_stable_and_keyed():
    a = src.fixed_noise((2, 19, 50), seed=4101)
    assert np.array_equal(a, src.fixed_noise((2, 19, 50), seed=4101))
    assert not np.array_equal(a, src.fixed_noise((2, 19, 51), seed=4101))


def test_ax_flow_collects_t1_mid_final_tokens_and_integrates():
    calls = []
    def step(x, t):
        calls.append(t)
        # Explicit shallow/middle/deep taps from the frozen expert callback.
        return np.ones_like(x), {"shallow": x + t, "middle": x + 2*t, "deep": x + 3*t}
    out, hidden = src.collect_ax_path(np.zeros((10, 7), np.float32), step, steps=2)
    assert np.allclose(out, -1)
    assert set(hidden) == {1.0, 0.5, 0.0}
    assert hidden[1.0]["deep"].shape == (10, 7)
    assert calls == [1.0, 0.5, 0.0]
    taps={t:{name:np.ones((10,8),np.float32) for name in src.AX_TAPS} for t in src.AX_TIMES}
    assert src.validate_ax_feature_grid(taps).shape==(3,3,10,8)


def test_ax_grid_refuses_endpoint_only_or_missing_layer_substitutes():
    endpoints={1.0:{name:np.ones((10,4)) for name in src.AX_TAPS},
               0.0:{name:np.ones((10,4)) for name in src.AX_TAPS}}
    with pytest.raises(ValueError,match="flow times"):
        src.validate_ax_feature_grid(endpoints)
    complete={t:{name:np.ones((10,4)) for name in src.AX_TAPS} for t in src.AX_TIMES}
    del complete[.5][src.AX_TAPS[1]]
    with pytest.raises(ValueError,match="expert layer taps"):
        src.validate_ax_feature_grid(complete)


def test_causal_pt_clip_rejects_future_and_requires_current_last():
    frames = np.zeros((16, 8, 8, 3), np.uint8)
    ids = np.arange(100, 116)
    assert src.validate_causal_clip(frames, current_frame_index=115, frame_indices=ids).shape[0] == 16
    with pytest.raises(ValueError, match="future"):
        src.validate_causal_clip(frames, current_frame_index=114, frame_indices=ids)
    with pytest.raises(ValueError, match="end at the query"):
        src.validate_causal_clip(frames, current_frame_index=116, frame_indices=ids)
    with pytest.raises(ValueError, match="contiguous"):
        src.validate_causal_clip(frames, current_frame_index=115,
                                 frame_indices=np.array([100, 101, 103] + list(range(104, 117))))


def test_track_tokens_encode_displacement_visibility_and_groups():
    tracks = np.zeros((4, 2, 2), np.float32)
    tracks[:, 0, 0] = np.arange(4)
    tracks[:, 1, 1] = np.arange(4) * 2
    vis = np.ones((4, 2), bool); vis[2, 1] = False
    tok, groups = src.tracks_to_tokens(tracks, vis, group=1, image_size=(4, 4))
    assert tok.shape == (2, 12) and groups.tolist() == [1, 1]
    assert tok.dtype == np.float16
    assert tok[0, 0] == 0 and tok[0, 3] == pytest.approx(.25)
    # Invisible tracker coordinates must not become displacement features.
    assert tok[1, 2 * 3] == 0 and tok[1, 2 * 3 + 1] == 0
    assert tok[1, 2*3+2] == 0


def test_cotracker_tensor_preserves_public_0_255_pixel_scale():
    frames = np.full((2, 4, 5, 3), 128, np.uint8)
    x = src.cotracker_video_tensor(frames)
    assert x.shape == (1, 2, 3, 4, 5)
    assert x.dtype == torch.float32 and torch.all(x == 128)


def test_same_shared_encoder_instance_encodes_both_sides():
    class Recording:
        def __init__(self): self.calls = []
        def __call__(self, **kw):
            self.calls.append(kw["tokens"])
            return kw["tokens"]
    enc = Recording()
    keys, queries = src.encode_key_query(enc, {"tokens": "memory"}, {"tokens": "query"})
    assert keys == "memory" and queries == "query"
    assert len(enc.calls) == 2


def test_dev_selection_is_top1_arm_mse_not_blend_metric():
    candidates = [
        {"setting": "top1-best", "top1_arm_mse": .10, "top20_blend_mse": .9},
        {"setting": "blend-best", "top1_arm_mse": .12, "top20_blend_mse": .1},
    ]
    assert src.choose_dev_setting(candidates)["setting"] == "top1-best"
    with pytest.raises(ValueError, match="locked"):
        src.choose_dev_setting(candidates, metric="top20_blend_mse")


def test_cache_round_trip_and_schema(tmp_path):
    path = tmp_path / "x.npz"
    keys = np.array([[1, 2, 3]], np.int64); tokens = np.ones((1, 4, 6), np.float16)
    groups = np.array([[0, 0, 1, 1]], np.int8)
    src.write_feature_cache(path, row_keys=keys, tokens=tokens, groups=groups,
                            source="test", config={"seed": 4101})
    got = src.load_feature_cache(path)
    assert np.array_equal(got[0], keys) and np.array_equal(got[1], tokens)
    assert np.array_equal(got[2], groups) and got[3] == "test" and got[4]["seed"] == 4101


def test_feature_cache_join_is_exact_and_preserves_requested_order():
    cache_keys=np.array([[1,2,30],[1,2,10],[1,2,20]])
    tok=np.arange(3*2*1).reshape(3,2,1); groups=np.zeros((3,2),np.int8)
    got,_=src.align_feature_cache([[1,2,10],[1,2,30]],cache_keys,tok,groups)
    assert np.array_equal(got[:,0,0],[2,0])
    with pytest.raises(KeyError):src.align_feature_cache([[1,2,40]],cache_keys,tok,groups)


def test_key2_injection_replaces_only_observations_and_keeps_eval_targets_separate():
    cache_keys=np.array([[3,4,5],[3,4,15]])
    cache=np.arange(2*3*2).reshape(2,3,2).astype(np.float16);groups=np.zeros((2,3),np.int8)
    raw={"train":{"key":cache_keys[:1],"actions":np.ones((1,10,7)),"vjepa_tokens":np.ones((1,1,2))},
         "eval":{"key":cache_keys[1:],"actions":np.zeros((1,10,7))}}
    fitted=src.inject_source_cache(raw,cache_keys=cache_keys,tokens=cache,groups=groups)
    assert np.array_equal(fitted['train']['tokens'],cache[:1])
    assert np.array_equal(fitted['train']['actions'],raw['train']['actions'])
    assert 'vjepa_tokens' not in fitted['train']
    assert np.array_equal(fitted['eval']['actions'],raw['eval']['actions'])


def test_lt1_conditions_use_shared_encoder_and_declared_history():
    assert src.key2_condition_config("E-AX")["encoder_input_mode"]=="e_vlm"
    assert src.key2_condition_config("E-PT+H")["encoder_input_mode"]=="e_vlm_h"
    assert src.key2_condition_config("E-AX+PT+H")["source"]=="ax_pt"
    key=np.array([[8,1,20]])
    ax=(key,np.ones((1,10,256),np.float16),np.full((1,10),2,np.int8))
    pt=(key,np.ones((1,100,48),np.float16),np.zeros((1,100),np.int8))
    data={'bank_input':{'key':key,'history':np.zeros((1,10,7)),'history_mask':np.ones((1,10)),
                        'state':np.zeros((1,8)),'tokens':np.zeros((1,1,2048)),
                        'vjepa_tokens':np.zeros((1,1,1024))}}
    combo=src.inject_condition_features(data,condition='E-AX+PT+H',ax_cache=ax,pt_cache=pt)['bank_input']
    assert combo['tokens'].shape==(1,10,256) and combo['vjepa_tokens'].shape==(1,100,48)
    pt_only=src.inject_condition_features(data,condition='E-PT',pt_cache=pt)['bank_input']
    assert pt_only['tokens'].shape==(1,100,48) and 'vjepa_tokens' not in pt_only


def test_key2_encoder_factory_reuses_shared_latent_encoder_class():
    sys.path.insert(0, str(HERE))
    import key_shared_latent
    model = src.make_shared_encoder(token_dim=12, with_history=False)
    assert isinstance(model, key_shared_latent.SharedLatentEncoder)
    assert model.token_dim == 12
    assert model.input_mode == "e_vlm"


def test_combined_condition_runs_pt_adapter_through_shared_key_query_encoder():
    model=src.make_condition_encoder("E-AX+PT+H")
    inputs={"tokens":torch.randn(2,6,256),"groups":torch.full((2,6),2),
            "history":torch.randn(2,10,7),"history_mask":torch.ones(2,10),
            "state":torch.randn(2,8),"vjepa_tokens":torch.randn(2,12,48),
            "vjepa_groups":torch.randint(0,2,(2,12))}
    key,query=src.encode_key_query(model,inputs,inputs)
    assert key.shape==query.shape==(2,5,256)
    assert model.pt_projection.proj[1].weight.grad is None
    (key.sum()+query.sum()).backward()
    assert model.pt_projection.proj[1].weight.grad is not None


def test_pt_only_condition_uses_pt_as_its_only_observation_source():
    model=src.make_condition_encoder("E-PT")
    out=model(tokens=torch.randn(1,20,48),groups=torch.zeros(1,20,dtype=torch.long))
    assert out.shape==(1,5,256)


def test_ax_pt_pack_combines_tokens_with_camera_groups():
    ax = torch.zeros(2, 10, 1024); ag = torch.zeros(2, 10, dtype=torch.long)
    pt = torch.zeros(2, 200, 48); pg = torch.cat([torch.zeros(2, 100), torch.ones(2, 100)], dim=1).long()
    projection = src.PointTokenProjection()
    tok, groups = src.pack_source_tokens(ax_tokens=ax, ax_groups=ag, pt_tokens=pt,
                                         pt_groups=pg, pt_projection=projection)
    assert tok.shape == (2, 210, 1024)
    assert groups.shape == (2, 210)
    assert groups[0, -100:].unique().tolist() == [1]
    ax_layer=torch.zeros(1,10,256); pt_layer=torch.zeros(1,200,48)
    proj_layer=src.PointTokenProjection(output_dim=256)
    packed,_=src.pack_source_tokens(ax_tokens=ax_layer,ax_groups=torch.full((1,10),2),
                                    pt_tokens=pt_layer,pt_groups=torch.zeros((1,200),dtype=torch.long),
                                    pt_projection=proj_layer)
    assert packed.shape==(1,210,256)


def test_feature_cache_refuses_accidental_overwrite(tmp_path):
    path = tmp_path / "cache.npz"
    kw = dict(row_keys=np.array([[1, 2, 3]]), tokens=np.zeros((1, 2, 4)),
              groups=np.zeros((1, 2), np.int8), source="x", config={})
    src.write_feature_cache(path, **kw)
    with pytest.raises(FileExistsError):
        src.write_feature_cache(path, **kw)
