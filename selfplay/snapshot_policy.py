"""Frozen SB3 PPO policies adapted to RLBOA negotiators."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from envs.rl_negotiator import PolicyRLBOANegotiator


SNAPSHOT_FORMAT_VERSION = 1


class FrozenPolicy:
    def __init__(self, policy, device="cpu"):
        self.policy = copy.deepcopy(policy).to(device)
        self.policy.set_training_mode(False)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    def predict(self, observation, state=None, deterministic=False):
        with torch.no_grad():
            return self.policy.predict(observation, state=state, deterministic=deterministic)


class SnapshotPolicyFactory:
    def __init__(self, checkpoint, device="cpu", deterministic=False):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.deterministic = deterministic
        metadata_path = self.checkpoint.with_suffix(".json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise ValueError("Unsupported RLBOA snapshot format")
        loaded = PPO.load(str(self.checkpoint), device=device)
        self.base_policy = FrozenPolicy(loaded.policy, device)

    def __call__(self, env, slot):
        metadata = self.metadata
        if env.domain_name not in metadata["compatible_domains"]:
            raise ValueError(f"Snapshot is incompatible with domain {env.domain_name}")
        if env.case not in metadata["compatible_cases"]:
            raise ValueError(f"Snapshot is incompatible with utility case {env.case}")
        if int(env.action_space.n) != int(metadata["action_count"]):
            raise ValueError("Snapshot action count does not match the environment")
        if list(env.observation_space.shape) != metadata["observation_shape"]:
            raise ValueError("Snapshot observation shape does not match the environment")
        return PolicyRLBOANegotiator(
            env.domain, FrozenPolicy(self.base_policy.policy, self.device),
            deterministic=self.deterministic, n_ranges=env.n_actions,
            entry_id=metadata["entry_id"], source="snapshot",
            name=f"{metadata['name']}-{slot}",
        )


class CurrentPolicyFactory:
    def __init__(self, model, device="cpu", deterministic=False):
        self.policy = FrozenPolicy(model.policy, device)
        self.device = device
        self.deterministic = deterministic

    def __call__(self, env, slot):
        return PolicyRLBOANegotiator(
            env.domain, FrozenPolicy(self.policy.policy, self.device),
            deterministic=self.deterministic, n_ranges=env.n_actions,
            entry_id="current-self-play", source="self_play",
            name=f"CurrentRLBOA-{slot}",
        )


def save_snapshot(model_dir, model, config, step, metrics):
    directory = Path(model_dir) / "pool" / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"snapshot-{int(step)}.zip"
    temporary = directory / f"snapshot-{int(step)}.tmp.zip"
    model.save(str(temporary))
    temporary.replace(destination)
    metadata = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "entry_id": f"snapshot-{int(step)}",
        "name": f"RLBOASnapshot{int(step)}",
        "global_step": int(step),
        "compatible_domains": list(config["issues"]),
        "compatible_cases": [config["case"]],
        "action_count": int(config["n_actions"]),
        "observation_shape": [7],
        "benchmark": metrics,
    }
    meta_destination = destination.with_suffix(".json")
    meta_temporary = meta_destination.with_suffix(".json.tmp")
    meta_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    meta_temporary.replace(meta_destination)
    return destination


def factory_for_entry(entry, model_dir, learner_model=None, device="cpu", deterministic=False):
    if entry.kind == "scripted":
        return entry.name
    if entry.kind == "snapshot":
        return SnapshotPolicyFactory(
            Path(model_dir) / entry.checkpoint_path, device=device, deterministic=deterministic
        )
    if entry.kind == "self_play":
        if learner_model is None:
            raise ValueError("Current self-play requires the learner model")
        return CurrentPolicyFactory(learner_model, device=device, deterministic=deterministic)
    raise ValueError(f"Unsupported pool entry kind: {entry.kind}")

