"""Atomic SB3 and self-play coordinator checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import cloudpickle
import numpy as np
import torch


FORMAT_VERSION = 1


def capture_random_states(context):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "context": context.state_dict(),
    }


def restore_random_states(states, context):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("cuda") is not None:
        torch.cuda.set_rng_state_all(states["cuda"])
    context.load_state_dict(states["context"])


def capture_environment(vec_env, model):
    states = []
    for environment in vec_env.envs:
        states.append({
            "rng_state": environment.rng_state(),
            "scenario_index": int(environment.scenario_index),
        })
    return {
        "envs": states,
        "last_obs": None if model._last_obs is None else model._last_obs.copy(),
        "last_episode_starts": (
            None if model._last_episode_starts is None
            else model._last_episode_starts.copy()
        ),
    }


def restore_environment(vec_env, model, saved, context, pool):
    if not saved or len(saved["envs"]) != len(vec_env.envs):
        raise ValueError("Checkpoint environment count does not match n_envs")
    for environment, payload in zip(vec_env.envs, saved["envs"]):
        # Version-1 checkpoints originally stored the complete NegMAS session.
        # Those third-party internals are not reliably pickle-restorable, so
        # retain only their RNG/scenario state and start a fresh episode.
        if isinstance(payload, (bytes, bytearray)):
            previous = cloudpickle.loads(payload)
            rng_state = previous.rng_state()
            scenario_index = previous.scenario_index
        else:
            rng_state = payload["rng_state"]
            scenario_index = payload["scenario_index"]
        environment.scenario_index = int(scenario_index)
        environment.set_rng_state(rng_state)
    model._last_obs = vec_env.reset()
    model._last_episode_starts = np.ones((len(vec_env.envs),), dtype=bool)


def save_checkpoint(model_dir, model, pool, config, context, schedule, vec_env=None):
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    temporary_model = root / "checkpoint.tmp.zip"
    model.save(str(temporary_model))
    temporary_model.replace(root / "checkpoint.zip")
    pool.save()
    state = {
        "format_version": FORMAT_VERSION,
        "global_step": int(model.num_timesteps),
        "cumulative_timesteps": int(model.num_timesteps),
        "config": dict(config),
        "observation_shape": [7], "action_count": int(config["n_actions"]),
        "domains": list(config["issues"]), "case": config["case"],
        "pool_metadata": pool.to_dict(), "schedule": dict(schedule),
        "random_states": capture_random_states(context),
        "environment": capture_environment(vec_env, model) if vec_env is not None else None,
    }
    temporary = root / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(root / "training_state.pt")
    return root / "checkpoint.zip"


def load_training_state(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        path = path / "training_state.pt"
    elif path.name == "checkpoint.zip":
        path = path.with_name("training_state.pt")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Torch 1.9 in the original RLBOA Docker image
        state = torch.load(path, map_location=device)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported training-state format")
    return state
