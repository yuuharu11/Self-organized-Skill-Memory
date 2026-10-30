"""Control budgets and initial-state selection of the public Skill evaluator."""
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]/'skill_moe/skillnet/examples/memory_pilot/anchor_flow'
sys.path.insert(0,str(HERE))
from run_case import environment_seed, initial_state_index, suite_control_limit, environment_horizon, episode_status


@pytest.mark.parametrize('suite,limit', [('libero_skill_obj',1600),('libero_skill',1600),
    ('libero_10',520),('libero_90',400),('libero_spatial',220),('libero_object',280),('libero_goal',300)])
def test_public_control_budgets(suite,limit):
    assert suite_control_limit(suite)==limit


def test_initial_states_are_public_prefix_not_wrapped():
    assert [initial_state_index(i,50) for i in range(20)]==list(range(20))
    with pytest.raises(IndexError):initial_state_index(50,50)
    with pytest.raises(IndexError):initial_state_index(-1,50)


def test_skill_environment_seed_matches_public_fixed_seed():
    assert [environment_seed('libero_skill_obj',7,i) for i in range(20)]==[7]*20
    assert environment_seed('libero_10',7,1)==8  # retain non-Skill paired protocol


def test_horizon_covers_full_control_and_stabilization():
    assert environment_horizon(1600,10)==1611
    assert environment_horizon(520,10)==1000
    assert environment_horizon(400,10)==1000


@pytest.mark.parametrize('success,returned_done,internal_done,expected',[
    (False,False,False,(False,False)),
    (False,False,True,(False,True)),  # LIBERO masks the robosuite timeout
    (False,True,False,(False,True)),
    (True,True,False,(True,True)),
])
def test_timeout_is_not_success(success,returned_done,internal_done,expected):
    from types import SimpleNamespace
    env=SimpleNamespace(env=SimpleNamespace(done=internal_done),check_success=lambda:success)
    assert episode_status(env,returned_done)==expected
