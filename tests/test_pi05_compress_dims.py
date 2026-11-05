import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT=Path(__file__).resolve().parents[1] / 'skill_moe/skillnet/examples/memory_pilot/pi05_memory/compress_vs_raw/dimension_extension.py'
spec=importlib.util.spec_from_file_location('pi05_compress_dimension_extension_test',SCRIPT)
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)


def test_codec_fit_split_excludes_validation_and_memory_but_documents_long_inclusion():
    rows={'validation':np.array([0,0,1,1,1],bool),'memory':np.array([0,1,0,0,0],bool),
          'task':np.array([0,3,1,10,35])}
    ids=module.codec_fit_ids(rows)
    np.testing.assert_array_equal(ids,[0])
    assert rows['task'][ids[0]]<10 # Long training examples are intentionally retained.
    hold=module.nonlong_holdout_ids(rows)
    np.testing.assert_array_equal(hold,[3,4])


def test_candidate_base_encoding_uses_dataset_row_ids():
    class Codec:
        def encode(self, actions, states):
            assert np.array_equal(states[:,0],actions[:,0,0]+50)
            return actions[:,0,:2], np.zeros((len(actions),4),np.float32)
    actions=np.zeros((20,10,32),np.float32); states=np.zeros((20,32),np.float32)
    actions[:,0,0]=np.arange(20); states[:,0]=np.arange(20)+50
    candidate_ids=np.array([[17,3],[11,1]])
    b=module.encode_candidate_b(Codec(),{'state':states},actions,candidate_ids)
    np.testing.assert_array_equal(b[:,:,0],candidate_ids)
    np.testing.assert_array_equal(b[:,:,1],np.zeros_like(candidate_ids))


def test_holm_step_down_is_monotone_and_bounded():
    adjusted=module._holm({'a':.001,'b':.02,'c':.04,'d':.9})
    assert all(0<=x<=1 for x in adjusted.values())
    assert adjusted['a']<=adjusted['b']<=adjusted['c']<=adjusted['d']


def test_reviewed_shuffle_is_exact_top1_permutation_and_candidates_are_memory_rows():
    fair=module.fair_module(); rows=module.base_module().load_rows(module.RUN)
    prep=np.load(module.REVIEW/'drafts_prepared.npz',allow_pickle=False)
    long=prep['eval_query_rows']; m2=prep['m2_candidate_rows']; shuffle=prep['shuffle_candidate_rows']
    assert np.array_equal(np.sort(m2),np.sort(shuffle))
    assert np.all(np.asarray(rows['task'][shuffle])!=np.asarray(rows['task'][long]))
    pool=np.asarray(np.load(module.MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'])
    assert np.all(np.isin(np.r_[m2,shuffle,prep['oracle_candidate_rows']],pool))
    assert len(fair.training_rows(rows))==11712


def test_e2_selected_candidate_ids_are_preserved_for_training_and_not_long():
    fair=module.fair_module(); rows=module.base_module().load_rows(module.RUN)
    saved=np.load(module.FAIR/'E-M2/training_drafts.npz',allow_pickle=False)
    assert np.array_equal(saved['query_rows'],fair.training_rows(rows))
    assert np.array_equal(saved['candidate_rows'][np.arange(len(saved['rank'])),saved['rank']],
                          saved['selected_candidate_rows'])
    pool=np.asarray(np.load(module.MEMORY/'units.npz',allow_pickle=False)['pool_row_ids'])
    fair.validate_candidates(saved['selected_candidate_rows'][:,None],saved['query_rows'],rows,pool)

