"""PFSP probability and snapshot score helpers."""

from __future__ import annotations

import math
import numpy as np


def pfsp_probabilities(entries, alpha=2.0, uniform_mix=0.1):
    if not entries:
        raise ValueError("PFSP needs at least one entry")
    if not math.isfinite(float(alpha)) or alpha < 0 or not 0 <= uniform_mix <= 1:
        raise ValueError("Invalid PFSP parameters")
    difficulties = np.asarray([entry.difficulty for entry in entries], dtype=np.float64)
    if np.any(~np.isfinite(difficulties)):
        raise ValueError("PFSP difficulty must be finite")
    logits = alpha * difficulties
    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()
    result = (1.0 - uniform_mix) * softmax + uniform_mix / len(entries)
    result /= result.sum()
    return result


def negotiation_score(agreement, utility, length, length_weight=-0.005, epsilon=0.01):
    agreement_rate = min(float(bool(agreement)), 1.0 - epsilon)
    return (1.0 - agreement_rate) ** (-float(utility)) + length_weight * float(length)

