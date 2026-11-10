import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'skill_moe/skillnet/examples/memory_pilot/unseen_libero/tasks.py'
S=importlib.util.spec_from_file_location('unseen_tasks',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def test_long_scene_matching_does_not_confuse_scene1_and_scene10():
    assert M.scene('KITCHEN_SCENE10_close_it')=='KITCHEN_SCENE10'
    assert 'KITCHEN_SCENE10' not in M.LONG_SCENES
    assert M.scene('LIVING_ROOM_SCENE1_put_it') in M.LONG_SCENES

def test_all_exclusions_apply_independently():
    rows=[{'task_id':0,'name':'KITCHEN_SCENE10_close_it','language':'close it'},
          {'task_id':1,'name':'KITCHEN_SCENE4_novel','language':'novel'},
          {'task_id':2,'name':'KITCHEN_SCENE9_turn_on_the_stove','language':'turn on the stove'},
          {'task_id':3,'name':'STUDY_SCENE3_move_book','language':' PICK_up  the BOOK '},
          {'task_id':4,'name':'STUDY_SCENE2_new','language':'new instruction'}]
    assert [x['task_id'] for x in M.eligible(rows,['pick up the book'])]==[0,4]

def test_scene_strata_deterministic_and_input_order_independent():
    rows=[{'task_id':10*s+i,'name':f'KITCHEN_SCENE{s+10}_move_{i}','language':f'move {s} {i}'} for s in range(11) for i in range(3)]
    a=M.select_tasks(rows,[]);b=M.select_tasks(rows[::-1],[])
    assert a==b and len(a)==10 and len({M.scene(x['name']) for x in a})==10
    assert not {M.scene(x['name']) for x in a}&M.LONG_SCENES

def test_actual_benchmark_has_no_training_or_long_overlap():
    # Require the real submodule: with examples/ on sys.path, bare 'libero'
    # resolves to an empty namespace stub and must not count as available.
    pytest.importorskip('libero.libero.benchmark')
    from libero.libero import benchmark
    rows=[];training=[]
    for name in ['libero_spatial','libero_object','libero_goal','libero_10']:
        suite=benchmark.get_benchmark_dict()[name]();training += [suite.get_task(i).language for i in range(suite.n_tasks)]
    suite=benchmark.get_benchmark_dict()['libero_90']()
    for i in range(suite.n_tasks):
        t=suite.get_task(i);rows.append({'task_id':i,'name':t.name,'language':t.language})
    overlap={x['name'] for x in rows if M.normalize_instruction(x['language']) in set(map(M.normalize_instruction,training))}
    assert overlap==M.EXPLICIT_OVERLAPS
    selected=M.select_tasks(rows,training)
    assert len(selected)==10
    assert all(M.normalize_instruction(x['language']) not in set(map(M.normalize_instruction,training)) for x in selected)
    assert not {M.scene(x['name']) for x in selected}&M.LONG_SCENES
