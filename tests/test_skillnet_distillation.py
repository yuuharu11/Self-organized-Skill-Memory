from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skill_moe/skillnet/examples/skill_conditioning'))
from distill_train import MatchedData
from distill_report import pair_counts


def dataset(tmp_path):
    directory=tmp_path/'data';directory.mkdir()
    rows=[]
    for kind,task,parent,seed in [('normal',1,'n1',0),('normal',2,'n2',0),('normal',6,'n6',0),('recovery',1,'p1',89),('recovery',2,'p2',73),('recovery',2,'p2',89),('recovery',6,'p6',89)]:
        path=directory/f'{kind}-{parent}-{seed}.npz'
        np.savez(path,actions=np.full((3,10,32),task,dtype=np.float32),state=np.zeros((3,32)),step=np.arange(3))
        rows.append(dict(id=path.stem,path=str(path),samples=3,kind=kind,task=task,parent=parent))
    (directory/'index.json').write_text(json.dumps(dict(shards=rows)))
    return MatchedData(tmp_path)


def test_common_normal_rows_and_task_frequency_are_identical(tmp_path):
    d=dataset(tmp_path)
    for update in range(30):
        a,ai=d.batch(update,'M0');b,bi=d.batch(update,'M1')
        assert ai[:6]==bi[:6]
        assert [r['task'] for r in ai]==[r['task'] for r in bi]
        assert all(r['shard'].startswith('normal') for r in ai)
        assert all(r['shard'].startswith('recovery') for r in bi[6:])
        assert len(ai)==len(bi)==8
        np.testing.assert_array_equal(a['actions'][:6],b['actions'][:6])


def test_two_seed_teacher_parent_is_not_duplicated_in_parent_schedule(tmp_path):
    d=dataset(tmp_path)
    assert d.parents==['p1','p2','p6']
    assert [len(d.recovery[p]) for p in d.parents]==[1,2,1]
    assert d.batch(11,'M1')[1]==d.batch(11,'M1')[1]


def test_incomplete_pairs_are_excluded_and_harm_not_hidden_by_new_success():
    rows=[dict(job=dict(case='a',task=0,condition='S0'),success=True),dict(job=dict(case='a',task=0,condition='M1'),success=False),dict(job=dict(case='b',task=1,condition='S0'),success=False),dict(job=dict(case='b',task=1,condition='M1'),success=True),dict(job=dict(case='c',task=2,condition='M1'),success=True)]
    p=pair_counts(rows,'S0','M1')
    assert (p['paired'],p['rescue'],p['harm'],p['net'])==(2,1,1,0)


def test_ar_mask_fix_preserves_unbatched_and_batched_semantics():
    import jax.numpy as jnp
    from distill_loss import joined_ar_mask
    prefix=jnp.array([0,0,0],bool);suffix=jnp.array([1,0],bool)
    expected=np.array([[0,0,0,1,0],[0,0,0,1,0]],bool)
    np.testing.assert_array_equal(joined_ar_mask(prefix,suffix,2),expected)
    np.testing.assert_array_equal(joined_ar_mask(jnp.stack([prefix,prefix]),suffix,2),expected)


def test_only_actually_executed_tail_is_allowed():
    from distill_data import executed_actions
    queries=[dict(step=125,actions=np.zeros((5,7)).tolist()),dict(step=130,actions=np.ones((5,7)).tolist())]
    actions=executed_actions(queries,125,132)
    assert actions.shape==(7,7)
    np.testing.assert_array_equal(actions[5:],np.ones((2,7)))
    assert not any(len(actions[i:i+10])==10 for i in range(0,len(actions),5))
    with pytest.raises(ValueError,match='noncontiguous'):
        executed_actions([queries[1]],125,132)


def test_saved_parameters_roundtrip_through_public_checkpoint_loader(tmp_path):
    from distill_train import save_public_parameters
    from openpi.models.model import restore_params
    tree={'layer':{'kernel':np.arange(6,dtype=np.float32).reshape(2,3)}}
    path=tmp_path/'params'
    save_public_parameters(path,tree)
    restored=restore_params(path,restore_type=np.ndarray)
    np.testing.assert_array_equal(restored['layer']['kernel'],tree['layer']['kernel'])
    assert restored['layer']['kernel'].dtype==np.float32
