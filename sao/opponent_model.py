from abc import ABCMeta, abstractmethod


class AbstractOpponentModel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, offer) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, offer, t) -> None:
        raise NotImplementedError


# No knowledge about the preference profile
class NoModel(AbstractOpponentModel):
    def __call__(self, offer):
        pass

    def update(self, offer, t):
        pass


# Complex Automated Negotiations: Theories, Models, and Software Competitions
# Learns the issue weights based on how often the value of an issue changes
# The value weights are estimated based on the frequency they are offered
class HardHeadedFrequencyModel(AbstractOpponentModel):
    def __init__(self, ufun, learn_coef=0.2, learn_value_addition=1):
        self.weights = {}
        self.evaluates = {}
        self.prevOffer = None
        self.amountOfIssues = len(ufun.weights)
        self.learnCoef = learn_coef
        self.learnValueAddition = learn_value_addition
        self.gamma = 0.25
        self.goldenValue = self.learnCoef / self.amountOfIssues
        if hasattr(ufun, "issue_utilities"):
            self.issues = None
            values = {key: value.mapping.keys() for key, value in ufun.issue_utilities.items()}
        else:
            self.issues = tuple(ufun.issues)
            values = {issue.name: tuple(issue) for issue in self.issues}
        for key, issue_values in values.items():
            self.weights[key] = 1.0 / self.amountOfIssues
            self.evaluates[key] = {value: 1.0 for value in issue_values}

    def _items(self, offer):
        if hasattr(offer, "items"):
            return list(offer.items())
        return [(issue.name, value) for issue, value in zip(self.issues, offer)]

    def __call__(self, offer):
        util = 0
        for k, v in self._items(offer):
            util += self.weights[k] * (self.evaluates[k][v] / max(self.evaluates[k].values()))
        return util

    def update(self, offer, t):
        if self.prevOffer is not None:
            last_diff = self.determine_difference(offer, self.prevOffer)
            num_of_unchanged = len(last_diff) - sum(last_diff)
            total_sum = 1 + self.goldenValue * num_of_unchanged
            maximum_weight = 1 - self.amountOfIssues * self.goldenValue / total_sum
            for k, i in zip(self.weights.keys(), last_diff):
                weight = self.weights[k]
                if i == 0 and weight < maximum_weight:
                    self.weights[k] = (weight + self.goldenValue) / total_sum
                else:
                    self.weights[k] = weight / total_sum

        for issue, evaluator in self._items(offer):
            self.evaluates[issue][evaluator] += self.learnValueAddition
        self.prevOffer = tuple(self._items(offer))

    def determine_difference(self, first, second):
        return [int(f == s) for f, s in zip(self._items(first), second)]


# Counts how often each value is offered
# The utility of a bid is the sum of the score of its values divided by the best possible score
# The model only uses the first 100 unique bids for its estimation
class CUHKAgentValueModel(AbstractOpponentModel):
    maximumBidsStored = 100

    def __init__(self, ufun):
        self.evaluates = {}
        self.bid_history = []
        self.maxPossibleTotal = 0
        if hasattr(ufun, "issue_utilities"):
            self.issues = None
            values = {key: value.mapping.keys() for key, value in ufun.issue_utilities.items()}
        else:
            self.issues = tuple(ufun.issues)
            values = {issue.name: tuple(issue) for issue in self.issues}
        for key, issue_values in values.items():
            self.evaluates[key] = {value: 0.0 for value in issue_values}

    def _mapping(self, offer):
        if hasattr(offer, "items"):
            return offer
        return {issue.name: value for issue, value in zip(self.issues, offer)}

    def __call__(self, offer):
        offer = self._mapping(offer)
        total_bid_value = 0.0
        for issue in self.evaluates.keys():
            v = offer[issue]
            counter_per_value = self.evaluates[issue][v]
            total_bid_value += counter_per_value
        if total_bid_value == 0:
            return 0.0
        return total_bid_value / self.maxPossibleTotal

    def update(self, offer, t):
        if len(self.bid_history) > self.maximumBidsStored:
            return
        if offer not in self.bid_history:
            self.bid_history.append(offer)
        if len(self.bid_history) <= self.maximumBidsStored:
            self.update_statistics(offer)

    def update_statistics(self, offer):
        offer = self._mapping(offer)
        for issue in self.evaluates.keys():
            v = offer[issue]
            if self.evaluates[issue][v] + 1 > max(self.evaluates[issue].values()):
                self.maxPossibleTotal += 1
            self.evaluates[issue][v] += 1


# Defines the opponent’s utility as one minus the agent’s utility
class OppositeModel(AbstractOpponentModel):
    def __init__(self, my_ufun):
        self.ufun = my_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass


# Perfect knowledge of the opponent’s preferences
class PerfectModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return self.ufun(offer)

    def update(self, offer, t):
        pass


# Defines the estimated utility as one minus the real utility
class WorstModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass
