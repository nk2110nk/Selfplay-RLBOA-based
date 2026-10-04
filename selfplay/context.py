"""Shared single-process state used by all DummyVecEnv workers."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .snapshot_policy import factory_for_entry


class SelfPlayContext:
    def __init__(self, pool, model_dir, seed=0, device="cpu", record_pool=True):
        self.pool = pool
        self.model_dir = Path(model_dir)
        self.rng = np.random.default_rng(int(seed))
        self.device = device
        self.record_pool = bool(record_pool)
        self.current_model = None
        self.source_selected = {"scripted": 0, "snapshot": 0, "self_play": 0}
        self.episodes = 0

    def sample_pair(self):
        entries = self.pool.sample_pair(
            self.rng, current_model_available=self.current_model is not None
        )
        for entry in entries:
            self.source_selected[entry.kind] += 1
        return entries

    def factories(self, entries, deterministic=False):
        return [
            factory_for_entry(
                entry, self.model_dir, learner_model=self.current_model,
                device=self.device, deterministic=deterministic,
            )
            for entry in entries
        ]

    def record_episode(self, entries, info):
        if self.record_pool:
            for entry in entries:
                if entry.kind != "self_play":
                    entry.record(info["agreement"], info["agent_utility"], info["length"])
        self.episodes += 1

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "source_selected": dict(self.source_selected),
            "episodes": int(self.episodes),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.source_selected.update(state.get("source_selected", {}))
        self.episodes = int(state.get("episodes", 0))

