import importlib.util
from pathlib import Path

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'skill_moe/skillnet/examples/memory_pilot/pi05_memory/skill_expert/run.py'
SPEC=importlib.util.spec_from_file_location('pi05_skill_expert_run',PATH)
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)

def test_training_mixture_has_fixed_categorical_contract():
    labels,counts=MOD.mix_categories(np.random.default_rng(20261104),100000)
    assert labels.shape==(100000,)
    rates=counts/counts.sum()
    np.testing.assert_allclose(rates,[.60,.15,.25],atol=.006)

def test_confidence_schedule_edges_and_fallback_are_fixed():
    edges={'q20':.2,'q60':.6,'q80':.8}
    assert MOD.confidence_mode(.19,edges,'C')=={'fallback':True,'t0':None,'draft':'normal'}
    assert MOD.confidence_mode(.2,edges,'C')['t0']==.6
    assert MOD.confidence_mode(.6,edges,'C')['t0']==.4
    assert MOD.confidence_mode(.8,edges,'C')['t0']==.2
    assert MOD.confidence_mode(.1,edges,'B')['t0']==.6
    assert MOD.confidence_mode(.1,edges,'A')['t0']==.6
    assert MOD.confidence_mode(.1,edges,'D')['draft']=='zero'

def test_fixed_score_edges_use_dev_only_quantiles():
    scores=np.arange(100,dtype=float)
    edges=MOD.confidence_edges(scores)
    assert edges['n_dev']==100
    np.testing.assert_allclose([edges['q20'],edges['q60'],edges['q80']],np.quantile(scores,[.2,.6,.8]))
    assert 'success' not in edges

def test_environment_and_query_seed_protocol():
    for init in range(10):
        env,q=MOD.seed_values(init,17)
        assert env==7+init
        assert q==(7+init)*100000+17

