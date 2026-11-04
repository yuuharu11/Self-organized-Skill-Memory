from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"skill_moe/skillnet/examples/skill_conditioning"))
from rescue_protocol import MAIN_CASES,action_prefix,base_prompts,main_decision,match_prompts,noise_seed,pair_table,teacher_action,teacher_translation_feasible


def audit(prompt):
    length=len(prompt.split())
    return dict(length=length,truncated=False,exact_encoding=True)


def test_prompt_padding_is_frozen_without_touching_original_or_residual():
    prompts=base_prompts("put both objects away","pick the bowl and put it on the plate")
    frozen=match_prompts(prompts,audit)
    assert frozen["matched_token_difference"]<=1
    assert frozen["prompts"]["A"]==prompts["A"]
    assert frozen["prompts"]["B"]==prompts["B"]
    for key in ("C","D"):
        assert frozen["prompts"][key].startswith(prompts[key])
    assert not prompts["C"].endswith("please")


def test_truncation_is_rejected_before_any_policy_outcome():
    def truncated(prompt):
        return dict(length=100,truncated=True,exact_encoding=True)
    with pytest.raises(ValueError,match="truncation"):
        match_prompts(base_prompts("whole task","remaining"),truncated)


def test_success_at_middle_of_last_recorded_chunk_preserves_only_executed_prefix():
    rows=[dict(step=0,actions=np.zeros((5,7)).tolist()),dict(step=5,actions=np.ones((5,7)).tolist())]
    prefix=action_prefix(rows,7)
    assert prefix.shape==(7,7)
    np.testing.assert_array_equal(prefix[5:],np.ones((2,7)))
    with pytest.raises(ValueError,match="noncontiguous"):
        action_prefix([rows[1]],7)


def test_noise_key_depends_on_absolute_control_time_not_query_count():
    five={step:noise_seed(73,20,step) for step in range(0,50,5)}
    two={step:noise_seed(73,20,step) for step in range(0,50,2)}
    assert all(five[step]==two[step] for step in set(five)&set(two))
    assert len({noise_seed(73,20,step) for step in range(800)})==800
    assert noise_seed(73,20,500)!=noise_seed(73,21,500)


def test_improving_only_on_a_broken_intervention_is_not_adopted():
    original={case:index<2 for index,case in enumerate(MAIN_CASES)}
    rows={(case,seed):{arm:dict(success=(arm in ("A","C"))) for arm in ("A","B","C","D")}
          for case in MAIN_CASES for seed in (73,89)}
    decision=main_decision(rows,original,MAIN_CASES)
    assert decision["C_vs_B"]["net"]>0
    assert decision["C_vs_A"]["net"]==0
    assert not decision["passes"]


def test_missing_arm_is_neither_failure_nor_an_extra_pair():
    rows={("task03-init20",73):dict(C=dict(success=True)),
          ("task07-init21",73):dict(A=dict(success=False),C=dict(success=True))}
    pair=pair_table(rows,"C","A")
    assert pair["n_seed_pairs"]==pair["rescue"]==1
    assert pair["n_parent_states"]==1


def test_teacher_is_a_bounded_real_action_and_does_not_modify_reference():
    reference=dict(eef=[.1,.2,.3],quat=[0.,0.,0.,1.],retreat_origin=[0.,0.,.3],grip=-1.)
    scale=np.asarray([.05,.05,.05,.5,.5,.5])
    no_rotation=lambda a,b:np.zeros(3)
    retreat=teacher_action([0.,0.,.3],[0,0,0,1],reference,0,scale,no_rotation)
    approach=teacher_action([0.,0.,.35],[0,0,0,1],reference,10,scale,no_rotation)
    assert retreat[2]>0 and approach[0]>0 and approach[1]>0
    assert np.max(np.abs(retreat[:6]))<=.5
    assert np.max(np.abs(approach[:6]))<=.5
    assert retreat[6]==approach[6]==-1
    assert reference["eef"]==[.1,.2,.3]


def test_unreachable_reference_is_not_scored_as_a_failed_teacher():
    assert not teacher_translation_feasible(dict(eef=[0,0,.9],retreat_origin=[0,0,1.3]))
    assert teacher_translation_feasible(dict(eef=[0,0,1.2],retreat_origin=[0,0,1.3]))


def test_adoption_needs_new_rescues_in_two_tasks_and_complete_matched_arms():
    original={case:index<2 for index,case in enumerate(MAIN_CASES)}
    rows={}
    for index,case in enumerate(MAIN_CASES):
        for seed in (73,89):
            rows[case,seed]={arm:dict(success=(index<2 or (arm=="C" and index in (2,3)))) for arm in ("A","B","C","D")}
    assert main_decision(rows,original,MAIN_CASES)["passes"]
    del rows[MAIN_CASES[2],73]["D"]
    assert not main_decision(rows,original,MAIN_CASES)["passes"]


def test_changing_target_in_a_window_is_not_contact_stall():
    from rescue_protocol import window_phenomena
    rows=[dict(target="bowl" if i<25 else "drawer",target_contact=True,
        progress_vector=[0.,0.,0.],command=[.5,0,0,0,0,0,-1],target_distance=.05,
        truth=dict(raw_goals=[False,False])) for i in range(50)]
    labels,_=window_phenomena(rows)
    assert "contact_stall" not in labels
