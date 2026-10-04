"""Balanced opponent-pool evaluation isolated from training RNG streams."""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch

from envs.compat import reset_result as _unused
from envs.env import RLBOAEnv
from results import row_from_info
from .entry import PoolEntry
from .pfsp import negotiation_score
from .snapshot_policy import factory_for_entry


UINT32_MODULUS = 2 ** 32


def evaluation_seed(base_seed, step, entry_index, episode_index):
    """Derive a reproducible NumPy-compatible seed from an evaluation event."""
    value = int(base_seed) + int(step) * 100003 + int(entry_index) * 1009 + int(episode_index)
    return value % UINT32_MODULUS


def balanced_schedule(domains, anchors, cases, minimum_episodes):
    base = list(itertools.product(domains, anchors, cases, (0, 1)))
    if not base or minimum_episodes < 1:
        raise ValueError("Evaluation requires a domain, anchor, case and episode")
    repeats = max(1, (minimum_episodes + len(base) - 1) // len(base))
    return (base * repeats)[:max(minimum_episodes, len(base))]


def _episode(model, entry, anchor, slot, domain, case, model_dir, seed,
             n_actions=10, device="cpu"):
    anchor_entry = PoolEntry(f"scripted-{anchor}", "scripted", anchor)
    entries = [anchor_entry, anchor_entry]
    entries[slot] = entry
    target = factory_for_entry(entry, model_dir, learner_model=model,
                               device=device, deterministic=True)
    factories = [anchor, anchor]
    factories[slot] = target
    env = RLBOAEnv(
        domain=domain, case=case, test=True, n_actions=n_actions,
        add_noise=False, random_train=False, fixed_entries=entries,
        fixed_factories=factories, model_dir=model_dir, seed=seed,
        record_pool=False,
    )
    reset = env.reset(seed=seed)
    observation = reset[0] if isinstance(reset, tuple) else reset
    done, info = False, None
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        result = env.step(action)
        if len(result) == 5:
            observation, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observation, _, done, info = result
    env.close()
    return info


def _aggregate(rows, length_weight):
    count = len(rows)
    result = {
        "agreement_rate": sum(row["agreement"] for row in rows) / count,
        "agent_utility": sum(row["agent_utility"] for row in rows) / count,
        "mean_length": sum(row["length"] for row in rows) / count,
    }
    result["negotiation_score"] = negotiation_score(
        result["agreement_rate"], result["agent_utility"],
        result["mean_length"], length_weight,
    )
    return result


def evaluate_pool(model, pool, domains, cases, episodes, model_dir, seed,
                  length_weight=-0.005, step=0, device="cpu", n_actions=10,
                  sampler_rng=None):
    anchors = [entry.name for entry in pool.entries if entry.kind == "scripted"]
    if not anchors:
        raise ValueError("Pool evaluation requires a scripted anchor")
    states = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": sampler_rng.bit_generator.state if sampler_rng is not None else None,
    }
    measurements, all_infos = {}, []
    schedule = balanced_schedule(domains, anchors, cases, episodes)
    try:
        for entry_index, entry in enumerate(list(pool.entries)):
            infos = []
            for episode_index, (domain, anchor, case, slot) in enumerate(schedule):
                episode_seed = evaluation_seed(seed, step, entry_index, episode_index)
                info = _episode(model, entry, anchor, slot, domain, case, model_dir,
                                episode_seed, n_actions, device)
                info["evaluation_seed"] = episode_seed
                info["evaluated_entry"] = entry.id
                info["evaluated_slot"] = slot
                infos.append(info)
                all_infos.append(info)
            measurements[entry.id] = _aggregate(infos, length_weight)
            for info in infos:
                if entry.kind != "self_play":
                    entry.record(info["agreement"], info["agent_utility"], info["length"])
        benchmark = _aggregate(all_infos, length_weight)
        rows = [row_from_info(info, info["evaluation_seed"]) for info in all_infos]
        return measurements, benchmark, rows
    finally:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["cuda"])
        if sampler_rng is not None:
            sampler_rng.bit_generator.state = states["sampler"]
