import random
import bisect
import numpy as np

from negmas.sao import SAONegotiator
from typing import Optional
from negmas.common import *
from negmas.outcomes import Outcome
try:
    from negmas.outcomes import ResponseType
except ImportError:  # negmas >= 0.9
    from negmas.sao import ResponseType
from sao.opponent_model import *
from envs.observer import *
from stable_baselines3 import PPO


class RLNegotiator(SAONegotiator):
    def __init__(self, name='RLAgent', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outcomes = None
        self.next_bid = None
        self.last_bid = None

    def on_ufun_changed(self):
        legacy_hook = getattr(super(), "on_ufun_changed", None)
        if legacy_hook is not None:
            legacy_hook()
        self.next_bid = None
        self.last_bid = None
        mechanism = getattr(self, "_ami", None) or getattr(self, "_nmi", None)
        self.n_outcomes = len(mechanism.discrete_outcomes())

    def on_preferences_changed(self, changes):
        hook = getattr(super(), "on_preferences_changed", None)
        if hook is not None:
            hook(changes)
        self.on_ufun_changed()

    @property
    def all_bids(self):
        return self._ami.discrete_outcomes()

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        self.last_bid = self.next_bid
        return self.next_bid

    def set_next_bid(self, next_bid) -> None:
        self.next_bid = next_bid


class TestRLNegotiator(RLNegotiator):
    def __init__(self, domain, path, opponent, deterministic=True, mode='issue', **kwargs):
        super().__init__(**kwargs)
        self.model = PPO.load(path)
        self.mode = mode    # issue, venas
        self.deterministic = deterministic
        self.observer = OnehotObserve2nT(domain, 6)
        self.domain = domain
        self.actions = None
        self.states = None
        self.opponent = opponent

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        # 初手AC用
        # if self.mode == 'issue':
        #     if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
        #         return ResponseType.END_NEGOTIATION
        # elif self.mode == 'venas':
        #     if self.actions == self.n_outcomes:
        #         return ResponseType.END_NEGOTIATION

        observation = self.observer(state.__dict__,self.opponent)
        self.actions, self.states = self.model.predict(observation, state=self.states, deterministic=self.deterministic)
        if self.mode == 'issue':
            if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER
        elif self.mode == 'venas':
            print(self.actions, self.n_outcomes)
            if self.actions == self.n_outcomes:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.actions is None:
            observation = self.observer(None,self.opponent)
            self.actions, self.states = self.model.predict(observation, state=self.states, deterministic=self.deterministic)
        if self.mode == 'issue':
            if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
                return None
            return {i.name: i.values[v] for i, v in zip(self.domain, self.actions)}
        elif self.mode == 'venas':
            if self.actions == self.n_outcomes:
                return None
            return self.all_bids[self.actions]


class RandomNegotiator(RLNegotiator):
    def __init__(self, name='Random', **kwargs):
        super().__init__(name=name, **kwargs)
        self.action = -1

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        self.action = random.randrange(self.n_outcomes + 1)
        if self.action == self.n_outcomes:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def reset(self):
        super().reset()

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.action == -1:
            self.action = random.randrange(self.n_outcomes + 1)
        if self.action == self.n_outcomes:
            # TODO: 初ターンAcceptへの対処
            return None
        return self.all_bids[self.action]


class RLBOANegotiator(RLNegotiator):
    def __init__(self, n_ranges=10, **kwargs):
        super().__init__(**kwargs)
        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        self.target = n_ranges - 1
        self.update_threshold = 1.1
        self.n_ranges = n_ranges
        self.range_index = None
        self.om = None   # if self._utility_function is None else NoModel()

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.target = self.n_ranges - 1
        self.om = HardHeadedFrequencyModel(self._utility_function)
        # self.om = NoModel()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
            # reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes])  # [::-1]])
        self.n_outcomes = len(self.ordered_utils)
        # 範囲のインデックスを取得
        step = np.linspace(self.ordered_utils[0], self.ordered_utils[-1], self.n_ranges+1)
        self.range_index = [0]
        for th in step[1:-1]:
            idx = bisect.bisect(self.ordered_utils, th)
            self.range_index.append(idx)
        self.range_index.append(len(self.ordered_utils))

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        # return ResponseType.REJECT_OFFER
        if state['relative_time'] < self.update_threshold:
            self.om.update(offer, state['relative_time'])
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        offered_util = self._utility_function(offer)
        if offered_util is None:
            return ResponseType.REJECT_OFFER
        my_util = self._utility_function(self.propose(state))
        if offered_util >= my_util and (self.reserved_value is None or offered_util > self.reserved_value):
            return ResponseType.ACCEPT_OFFER
        if self.reserved_value is not None and my_util < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if type(self.om) == NoModel:
            return random.choice(self.get_bid_range())[1]
        else:
            return self.get_best_bid(self.get_bid_range())

    def get_best_bid(self, bids):
        best_util = -1
        _, best_bid = bids[0]
        all_zero = True
        for (u, o) in bids:
            evaluation = self.om(o)
            if evaluation > 0.0001:
                all_zero = False
            if evaluation > best_util:
                best_bid = o
                best_util = evaluation
        if all_zero:
            _, best_bid = random.choice(bids)
        return best_bid

    def get_bid_range(self):
        bids = self.ordered_outcomes[self.range_index[self.target]:self.range_index[self.target+1]]
        if bids:
            return bids

        for offset in range(1, self.n_ranges):
            lower = self.target - offset
            if lower >= 0:
                bids = self.ordered_outcomes[self.range_index[lower]:self.range_index[lower+1]]
                if bids:
                    return bids
            upper = self.target + offset
            if upper < self.n_ranges:
                bids = self.ordered_outcomes[self.range_index[upper]:self.range_index[upper+1]]
                if bids:
                    return bids
        return self.ordered_outcomes

    def set_target(self, target) -> None:
        self.target = min(max(int(target), 0), self.n_ranges - 1)


class TestRLBOANegotiator(RLBOANegotiator):
    def __init__(self, domain, path, opponent, deterministic=True, n_ranges=10, **kwargs):
        super().__init__(n_ranges=n_ranges, **kwargs)
        self.model = PPO.load(path)
        self.domain = domain
        self.opponent = opponent
        self.deterministic = deterministic
        self.observer = None
        self.states = None
        self.actions = None

    def on_ufun_changed(self):
        opponent_ids = () if self.observer is None else self.observer.opponent_ids
        super().on_ufun_changed()
        self.observer = RLBOAObserve(
            self.domain, self._utility_function, own_id=self.id,
            opponent_ids=opponent_ids,
        )
        self.observer.reset()

    def configure_observer_roles(self, opponent_ids):
        if self.observer is None:
            self.observer = RLBOAObserve(self.domain, self._utility_function, own_id=self.id)
        self.observer.configure_roles(self.id, opponent_ids)
        self.observer.reset()

    def _predict_target(self, state):
        if self.observer is None:
            self.observer = RLBOAObserve(self.domain, self._utility_function, own_id=self.id)
            self.observer.reset()
        state_dict = None if state is None else state.__dict__
        observation = self.observer(state_dict, self.opponent)
        self.actions, self.states = self.model.predict(
            observation,
            state=self.states,
            deterministic=self.deterministic,
        )
        self.set_target(self.actions)

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        self._predict_target(state)
        return super().respond(state, offer)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.actions is None:
            self._predict_target(None)
        return super().propose(state)


class PolicyRLBOANegotiator(RLBOANegotiator):
    """RLBOA opponent driven by an immutable SB3 policy copy."""

    def __init__(self, domain, policy, deterministic=False, n_ranges=10,
                 entry_id="snapshot", source="snapshot", **kwargs):
        super().__init__(n_ranges=n_ranges, **kwargs)
        self.domain = domain
        self.policy = policy
        self.deterministic = bool(deterministic)
        self.entry_id = entry_id
        self.source = source
        self.observer = None
        self.states = None
        self.actions = None

    def on_ufun_changed(self):
        opponent_ids = () if self.observer is None else self.observer.opponent_ids
        super().on_ufun_changed()
        self.observer = RLBOAObserve(
            self.domain, self._utility_function, own_id=self.id,
            opponent_ids=opponent_ids,
        )
        self.observer.reset()

    def configure_observer_roles(self, opponent_ids):
        if self.observer is None:
            self.observer = RLBOAObserve(self.domain, self._utility_function, own_id=self.id)
        self.observer.configure_roles(self.id, opponent_ids)
        self.observer.reset()

    @staticmethod
    def _state_dict(state):
        if state is None:
            return None
        if hasattr(state, "asdict"):
            return state.asdict()
        return getattr(state, "__dict__", state)

    def _predict_target(self, state):
        if self.observer is None:
            raise RuntimeError("Policy opponent observer is not initialized")
        observation = self.observer(self._state_dict(state))
        action, self.states = self.policy.predict(
            observation, state=self.states, deterministic=self.deterministic
        )
        self.actions = action
        self.set_target(action)

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        self._predict_target(state)
        return super().respond(state, offer)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.actions is None:
            self._predict_target(None)
        return super().propose(state)
