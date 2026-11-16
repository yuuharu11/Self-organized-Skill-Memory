import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skill_moe/skillnet/examples/memory_pilot/unseen_libero'))
import key_grid as kg


def test_vjepa_base_wrist_are_separate_concatenated_means():
    tok=np.zeros((2,4,1024),np.float32);groups=np.asarray([[0,0,1,1],[0,1,1,-1]])
    tok[0,:2]=2;tok[0,2:]=3;tok[1,0]=4;tok[1,1:3]=5
    x=kg._vjepa_two_camera(tok,groups)
    assert x.shape==(2,2048)
    assert np.all(x[0,:1024]==2) and np.all(x[0,1024:]==3)
    assert np.all(x[1,:1024]==4) and np.all(x[1,1024:]==5)


def test_vector_features_obey_requested_concat_inputs():
    part={'tokens':np.ones((3,5,2048),np.float32),'groups':np.zeros((3,5),np.int8),
      'history':np.ones((3,10,7),np.float32),'history_mask':np.ones((3,10),bool),
      'state':np.ones((3,8),np.float32),'vjepa_tokens':np.ones((3,4,1024),np.float32),
      'vjepa_groups':np.asarray([[0,0,1,1]]*3)}
    assert kg.make_vector_features(part,'vlm').shape==(3,2048)
    assert kg.make_vector_features(part,'history_state').shape==(3,88)
    assert kg.make_vector_features(part,'vlm_history_state').shape==(3,2136)
    assert kg.make_vector_features(part,'vjepa_history_state').shape==(3,2136)
    assert kg.make_vector_features(part,'vjepa_vlm_history_state').shape==(3,4184)


def test_pair_mlp_shape_and_shared_parameter_tying():
    shared=kg.PairMLPRanker(88,shared=True)
    separate=kg.PairMLPRanker(88,shared=False)
    assert shared.query_tower is shared.key_tower
    assert separate.query_tower is not separate.key_tower
    q=torch.randn(4,88);k=torch.randn(9,88)
    # Untied query is a single VectorRanker-style vector repeated across the
    # five key tokens; tied mode applies the same token-producing MLP to both.
    qrep=separate.query(q)
    assert qrep.shape==(4,5,256)
    assert torch.equal(qrep[:,0],qrep[:,4])
    scores=shared(q,k)
    assert scores.shape==(4,9) and torch.isfinite(scores).all()
    assert separate(q,k).shape==(4,9)


def test_input_rows_and_existing_e_conditions_match_keygrid_table():
    assert tuple(kg.INPUTS)==('vlm','history_state','vlm_history_state','vjepa_history_state','vjepa_vlm_history_state')
    assert kg.EXISTING_E['history_state']=='e_h'
    assert kg.EXISTING_E['vjepa_history_state']=='e_vj_h'
    assert kg.EXISTING_E['vjepa_vlm_history_state']=='e_vj_vlm_h'


def test_candidate_positive_indices_join_by_identity_not_array_order():
    q=np.asarray([[8,2,30],[4,1,10],[8,2,40]],np.int32)
    c=np.asarray([[4,1,10],[8,2,40],[8,2,30]],np.int32)
    assert np.array_equal(kg.exact_candidate_row_ids(q,c),[2,0,1])


def test_candidate_positive_mapping_rejects_missing_or_duplicate_keys():
    import pytest
    with pytest.raises(KeyError):kg.exact_candidate_row_ids([[1,2,3]],[[1,2,4]])
    with pytest.raises(ValueError):kg.exact_candidate_row_ids([[1,2,3]],[[1,2,3],[1,2,3]])
