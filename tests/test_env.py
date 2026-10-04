import numpy as np
import pytest

from envs.env import CASE_ORDERS, RLBOAEnv
from envs.observer import RLBOAObserve


@pytest.mark.parametrize("case,order", CASE_ORDERS.items())
def test_utility_case_assignment(case, order):
    env = RLBOAEnv(domain="Laptop", opponent=["Boulware", "Linear"],
                   case=case, add_noise=False)
    assert env.my_util is env.utilities[order[0]]
    assert env.opp_util1 is env.utilities[order[1]]
    assert env.opp_util2 is env.utilities[order[2]]
    env.close()


def test_observer_reset_is_complete_and_independent():
    observer = RLBOAObserve([], lambda offer: offer["u"], "me", ("a", "b"))
    first = observer.reset()
    observer.observe({"current_offer": {"u": .8}, "current_proposer": "a",
                      "relative_time": .2})
    second = observer.reset()
    assert np.array_equal(second, np.zeros(7, dtype=np.float32))
    assert first is not second and not np.shares_memory(first, second)


def _run(env, action=9):
    reset = env.reset()
    decisions = 0
    while True:
        result = env.step(action)
        decisions += 1
        observation, reward, terminated, truncated, info = result
        if terminated or truncated:
            return decisions, info
        assert env._next_negotiator_is_rl()


def test_learner_turn_boundary_and_terminal_info():
    env = RLBOAEnv(domain="Laptop", opponent=["Boulware", "Conceder"],
                   case="case2", test=True, add_noise=False, seed=2)
    decisions, info = _run(env)
    assert decisions <= 80
    required = {"domain", "case", "opponent1_id", "opponent2_id",
                "opponent1_source", "opponent2_source", "agent_utility",
                "opponent1_utility", "opponent2_utility", "agreement",
                "length", "state"}
    assert required <= info.keys()
    assert info["case"] == "case2"
    env.close()


@pytest.mark.parametrize("name", ["Boulware", "Linear", "Conceder", "Atlas3", "CUHKAgent"])
def test_supported_scripted_opponents(name):
    env = RLBOAEnv(domain="Laptop", opponent=[name, name], test=True,
                   add_noise=False, seed=1)
    env.reset()
    result = env.step(9)
    assert len(result) in (4, 5)
    env.close()


def test_seed_reproducibility():
    def trace():
        env = RLBOAEnv(domain="Laptop", opponent=["Boulware", "Conceder"],
                       test=True, add_noise=False, seed=33)
        _, info = _run(env)
        env.close()
        return info["agreement"], info["length"], info["agent_utility"]
    assert trace() == trace()
