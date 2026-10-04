"""Three-party RLBOA Gym environment with opponent-pool self play."""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch

from .compat import gym, reset_result, step_result
from .domain_loader import load_genius_domain
from .observer import RLBOAObserve
from .rl_negotiator import RLBOANegotiator
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import (
    AgentGG, AgentK, Atlas3, AverageTitForTatNegotiator, CUHKAgent,
    HardHeaded, TimeBasedNegotiator,
)
from selfplay.entry import PoolEntry
from selfplay.snapshot_policy import factory_for_entry


PENALTY = -1.0
CASE_ORDERS = {
    "case1": (0, 1, 2),
    "case2": (1, 2, 0),
    "case3": (2, 0, 1),
}


def _state_dict(state):
    if state is None or isinstance(state, dict):
        return state
    if hasattr(state, "asdict"):
        return state.asdict()
    return state.__dict__


class _LegacyNMI:
    """Expose dict outcomes expected by the NegMAS-0.8 baseline agents."""
    def __init__(self, nmi):
        self._nmi = nmi

    def discrete_outcomes(self):
        outcomes = self._nmi.discrete_outcomes()
        issues = self._nmi.issues
        return [outcome if hasattr(outcome, "items") else
                {issue.name: value for issue, value in zip(issues, outcome)}
                for outcome in outcomes]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_nmi"), name)


class _LegacyUtility:
    def __init__(self, utility):
        self._utility = utility

    def __call__(self, outcome):
        if hasattr(outcome, "items"):
            outcome = tuple(outcome[issue.name] for issue in self._utility.issues)
        return self._utility(outcome)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_utility"), name)


def _utility_value(utility, outcome):
    if outcome is not None and hasattr(outcome, "items") and hasattr(utility, "issues"):
        outcome = tuple(outcome[issue.name] for issue in utility.issues)
    return utility(outcome)


class NaiveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain="party", opponent=None, case="case1", test=False,
                 random_train=True, seed=0, **kwargs):
        super().__init__()
        if case not in CASE_ORDERS:
            raise ValueError(f"case must be one of {tuple(CASE_ORDERS)}")
        self.test = bool(test)
        self.case = case
        self.domain_names = self._normalize_domains(domain)
        self.opponent_pairs = self._normalize_opponent_pairs(opponent)
        self.scenarios = list(itertools.product(self.domain_names, self.opponent_pairs))
        self.random_train = bool(random_train)
        self.scenario_index = -1
        self.my_agent = self.session = None
        self.domain_name = self.domain = self.utilities = None
        self.util1 = self.util2 = self.util3 = None
        self.my_util = self.opp_util1 = self.opp_util2 = None
        self.opponent = self.state = self.observation = None
        self.reward_range = (PENALTY, 1.0)
        self._seed = 0
        self.np_random = None
        self.seed(seed)
        self._select_scenario()
        self.scenario_index = -1

    @staticmethod
    def _normalize_domains(domain):
        values = [domain] if isinstance(domain, str) else list(domain)
        if not values:
            raise ValueError("At least one domain is required")
        return values

    @staticmethod
    def _normalize_opponent_pairs(opponent):
        if opponent is None:
            return [["Boulware", "Conceder"]]
        if isinstance(opponent, str):
            return [[opponent, opponent]]
        opponent = list(opponent)
        if len(opponent) == 2 and all(isinstance(value, str) for value in opponent):
            return [opponent]
        pairs = []
        for value in opponent:
            pair = [value, value] if isinstance(value, str) else list(value)
            if len(pair) != 2:
                raise ValueError("Each opponent setting must contain two opponents")
            pairs.append(pair)
        if not pairs:
            raise ValueError("At least one opponent setting is required")
        return pairs

    def seed(self, seed=None):
        if seed is None:
            seed = self._seed
        self._seed = int(seed)
        self.np_random = np.random.default_rng(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        if hasattr(self, "action_space"):
            self.action_space.seed(self._seed)
        if hasattr(self, "observation_space"):
            self.observation_space.seed(self._seed)
        return [self._seed]

    def rng_state(self):
        return self.np_random.bit_generator.state

    def set_rng_state(self, state):
        self.np_random.bit_generator.state = state

    def _cleanup_ufuns(self):
        for ufun in (self.my_util, self.opp_util1, self.opp_util2):
            if ufun is not None and hasattr(ufun, "_ami"):
                del ufun._ami

    def _select_scenario(self):
        if self.random_train and len(self.scenarios) > 1:
            index = int(self.np_random.integers(len(self.scenarios)))
            domain_name, opponent = self.scenarios[index]
        else:
            self.scenario_index = (self.scenario_index + 1) % len(self.scenarios)
            domain_name, opponent = self.scenarios[self.scenario_index]
        self.domain_name = domain_name
        self.domain, utilities = load_genius_domain(domain_name)
        self.utilities = tuple(utilities)
        self.util1, self.util2, self.util3 = self.utilities
        order = CASE_ORDERS[self.case]
        self.my_util, self.opp_util1, self.opp_util2 = [self.utilities[i] for i in order]
        self.opponent = list(opponent)

    def make_scripted_opponent(self, name, slot, add_noise=False):
        kwargs = {"name": f"{name}-{slot}", "add_noise": add_noise}
        if name == "Boulware":
            return TimeBasedNegotiator(aspiration_type=10.0, **kwargs)
        if name == "Linear":
            return TimeBasedNegotiator(aspiration_type=1.0, **kwargs)
        if name == "Conceder":
            return TimeBasedNegotiator(aspiration_type=0.2, **kwargs)
        if name == "TitForTat1":
            return AverageTitForTatNegotiator(gamma=1, **kwargs)
        if name == "TitForTat2":
            return AverageTitForTatNegotiator(gamma=2, **kwargs)
        classes = {"AgentK": AgentK, "HardHeaded": HardHeaded,
                   "CUHKAgent": CUHKAgent, "Atlas3": Atlas3, "AgentGG": AgentGG}
        if name in classes:
            return classes[name](**kwargs)
        raise ValueError(f"Unknown scripted opponent: {name}")

    def get_reward(self):
        if self.state["timedout"] or self.state["broken"]:
            return 0.0 if self.test else PENALTY
        if self.state["agreement"] is not None:
            return float(_utility_value(self.my_util, self.state["agreement"]))
        return 0.0

    def close(self):
        self._cleanup_ufuns()
        if self.session is not None:
            try:
                self.session.reset()
            except (AttributeError, RuntimeError):
                pass
        self.session = self.my_agent = None


class RLBOAEnv(NaiveEnv):
    """One Gym step equals exactly one learner decision."""

    def __init__(self, domain="party", opponent=None, case="case1", test=False,
                 n_actions=10, add_noise=True, random_train=True,
                 selfplay_context=None, fixed_entries=None, fixed_factories=None,
                 model_dir=None, record_pool=True, seed=0, **kwargs):
        super().__init__(domain, opponent, case=case, test=test,
                         random_train=random_train, seed=seed, **kwargs)
        self.n_actions = int(n_actions)
        self.add_noise = bool(add_noise)
        self.selfplay_context = selfplay_context
        self.fixed_entries = tuple(fixed_entries) if fixed_entries else None
        self.fixed_factories = tuple(fixed_factories) if fixed_factories else None
        self.model_dir = model_dir
        self.record_pool = bool(record_pool)
        self.current_entries = self.opponents = None
        self.observer = RLBOAObserve(
            self.domain, lambda outcome: _utility_value(self.my_util, outcome)
        )
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

    def _entries_and_factories(self):
        if self.fixed_entries is not None:
            if self.fixed_factories is not None:
                return self.fixed_entries, self.fixed_factories
            factories = tuple(
                factory_for_entry(entry, self.model_dir,
                                  learner_model=getattr(self.selfplay_context, "current_model", None))
                for entry in self.fixed_entries
            )
            return self.fixed_entries, factories
        if self.selfplay_context is not None:
            entries = self.selfplay_context.sample_pair()
            return entries, tuple(self.selfplay_context.factories(entries))
        entries = tuple(PoolEntry(f"scripted-{name}", "scripted", name)
                        for name in self.opponent)
        return entries, tuple(entry.name for entry in entries)

    def _make_opponent(self, factory, slot):
        if isinstance(factory, str):
            return self.make_scripted_opponent(factory, slot, self.add_noise)
        return factory(self, slot)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._cleanup_ufuns()
        if self.session is not None:
            try:
                self.session.reset()
            except (AttributeError, RuntimeError):
                pass
        self._select_scenario()
        self.observer = RLBOAObserve(
            self.domain, lambda outcome: _utility_value(self.my_util, outcome)
        )
        self.observation_space = self.observer.observation_space
        self.session = MySAOMechanism(issues=self.domain, n_steps=80,
                                      avoid_ultimatum=False)
        self.my_agent = RLBOANegotiator(n_ranges=self.n_actions, name="RLAgent")
        entries, factories = self._entries_and_factories()
        opponents = [self._make_opponent(factory, slot) for slot, factory in enumerate(factories)]
        self.session.add(self.my_agent, ufun=self.my_util)
        self.session.add(opponents[0], ufun=self.opp_util1)
        self.session.add(opponents[1], ufun=self.opp_util2)
        # NegMAS 0.8 invokes this hook itself.  The compatibility environment
        # used for tests is 0.9, where the names changed to NMI/preferences.
        for negotiator in self.session.negotiators:
            if not hasattr(negotiator, "_ami"):
                negotiator._ami = _LegacyNMI(negotiator._nmi)
            if not hasattr(negotiator, "_utility_function"):
                negotiator._utility_function = _LegacyUtility(negotiator._preferences)
            if not hasattr(negotiator, "utility_function"):
                negotiator.utility_function = negotiator._utility_function
            if getattr(negotiator, "ordered_outcomes", None) is None:
                negotiator.on_ufun_changed()
        ids = [negotiator.id for negotiator in self.session.negotiators]
        self.observer.configure_roles(ids[0], ids[1:])
        for index, opponent in enumerate(opponents, start=1):
            if hasattr(opponent, "configure_observer_roles"):
                opponent.configure_observer_roles([ids[i] for i in range(3) if i != index])
        self.current_entries = entries
        self.opponents = tuple(opponents)
        self.state = None
        self.observation = self.observer.reset()
        return reset_result(self.observation.copy(), {
            "domain": self.domain_name, "case": self.case,
            "opponent_ids": [entry.id for entry in entries],
        })

    def step(self, action):
        self.my_agent.set_target(int(np.asarray(action).item()))
        while True:
            self.state = _state_dict(self.session.step())
            self.observation = self.observer(self.state)
            terminal = (self.state["agreement"] is not None or
                        self.state["timedout"] or self.state["broken"])
            if terminal or self._next_negotiator_is_rl():
                break
        reward = self.get_reward()
        info = self._terminal_info() if terminal else {}
        if terminal and self.selfplay_context is not None and self.record_pool:
            self.selfplay_context.record_episode(self.current_entries, info)
        terminated = bool(self.state["agreement"] is not None or self.state["broken"])
        truncated = bool(self.state["timedout"])
        return step_result(self.observation.copy(), reward, terminated, truncated, info)

    def _terminal_info(self):
        agreement = self.state["agreement"]
        agreed = agreement is not None
        utilities = [float(_utility_value(ufun, agreement)) if agreed else 0.0
                     for ufun in (self.my_util, self.opp_util1, self.opp_util2)]
        return {
            "domain": self.domain_name, "case": self.case,
            "opponent1_id": self.current_entries[0].id,
            "opponent2_id": self.current_entries[1].id,
            "opponent1_source": self.current_entries[0].kind,
            "opponent2_source": self.current_entries[1].kind,
            "agent_utility": utilities[0],
            "opponent1_utility": utilities[1],
            "opponent2_utility": utilities[2],
            "agreement": int(agreed), "length": int(self.state.get("step", 0)),
            "state": dict(self.state),
        }

    def _next_negotiator_is_rl(self):
        negotiators = self.session.negotiators
        next_index = (self.session._last_checked_negotiator + 1) % len(negotiators)
        return negotiators[next_index] is self.my_agent
