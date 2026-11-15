"""Outcome-independent scene-stratified LIBERO-90 inventory."""
import re
import numpy as np

SEED = 20261106
LONG_SCENES = frozenset(['KITCHEN_SCENE3','KITCHEN_SCENE4','KITCHEN_SCENE6','KITCHEN_SCENE8',
 'LIVING_ROOM_SCENE1','LIVING_ROOM_SCENE2','LIVING_ROOM_SCENE5','LIVING_ROOM_SCENE6','STUDY_SCENE1'])
EXPLICIT_OVERLAPS = frozenset(['KITCHEN_SCENE3_turn_on_the_stove','KITCHEN_SCENE9_turn_on_the_stove',
 'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'])

def normalize_instruction(text):
    return ' '.join(text.lower().replace('_',' ').split())

def scene(name):
    match = re.match(r'^((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)(?:_|$)', name)
    if not match: raise ValueError(name)
    return match.group(1)

def eligible(rows, training_instructions):
    seen = {normalize_instruction(x) for x in training_instructions}
    return [x for x in rows if x['name'] not in EXPLICIT_OVERLAPS
            and scene(x['name']) not in LONG_SCENES
            and normalize_instruction(x['language']) not in seen]

def select_tasks(rows, training_instructions, n=10, seed=SEED):
    pool = eligible(rows, training_instructions)
    groups = {}
    for x in sorted(pool, key=lambda x:x['name']): groups.setdefault(scene(x['name']),[]).append(x)
    rng=np.random.default_rng(seed)
    order=list(rng.permutation(sorted(groups)))
    selected=[]
    while len(selected)<n:
        changed=False
        for key in order:
            if not groups[key]:continue
            selected.append(groups[key].pop(int(rng.integers(len(groups[key])))))
            changed=True
            if len(selected)==n:break
        if not changed:raise ValueError('not enough eligible tasks')
    return sorted(selected,key=lambda x:x['task_id'])
