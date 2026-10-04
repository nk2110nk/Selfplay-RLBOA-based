"""Persistent ordered-pair opponent pool with source-mixture PFSP."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .entry import PoolEntry
from .pfsp import pfsp_probabilities


POOL_FORMAT_VERSION = 1


def is_dominant(current, incumbent, tolerance=0.01, min_agreement=0.0):
    if current["agreement_rate"] < min_agreement:
        return False
    delta = current["agent_utility"] - incumbent.get("agent_utility", 0.0)
    if delta > tolerance:
        return True
    if delta < -tolerance:
        return False
    return current["negotiation_score"] > incumbent.get("negotiation_score", float("-inf"))


def snapshot_pruning_key(entry):
    return entry.difficulty, entry.selected, entry.added_step


class OpponentPool:
    def __init__(self, root, scripted_names, max_size=16, pfsp_alpha=2.0,
                 uniform_mix=0.1, allow_duplicates=True,
                 self_play_probability=0.0, scripted_probability=0.5,
                 snapshot_probability=0.5, path=None, model_root=None):
        self.root = Path(root)
        self.path = Path(path) if path else self.root / "pool.json"
        self.model_root = Path(model_root) if model_root else self.root.parent
        self.max_size = int(max_size)
        self.pfsp_alpha = float(pfsp_alpha)
        self.uniform_mix = float(uniform_mix)
        self.allow_duplicates = bool(allow_duplicates)
        self.source_probabilities = {
            "self_play": float(self_play_probability),
            "scripted": float(scripted_probability),
            "snapshot": float(snapshot_probability),
        }
        weights = np.asarray(list(self.source_probabilities.values()), dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Source probabilities must be finite, non-negative, and not all zero")
        self.entries = [
            PoolEntry(f"scripted-{name}", "scripted", name)
            for name in dict.fromkeys(scripted_names)
        ]

    def _flattened_distribution(self, current_model_available=False):
        groups = []
        if current_model_available and self.source_probabilities["self_play"] > 0:
            groups.append((self.source_probabilities["self_play"], [
                PoolEntry("current-self-play", "self_play", "CurrentRLBOA")
            ]))
        scripted = [entry for entry in self.entries if entry.kind == "scripted"]
        snapshots = [entry for entry in self.entries if entry.kind == "snapshot"]
        if scripted and self.source_probabilities["scripted"] > 0:
            groups.append((self.source_probabilities["scripted"], scripted))
        if snapshots and self.source_probabilities["snapshot"] > 0:
            groups.append((self.source_probabilities["snapshot"], snapshots))
        total = sum(weight for weight, _ in groups)
        if total <= 0:
            raise ValueError("No available opponent source")
        entries, weights = [], []
        for source_weight, candidates in groups:
            within = (
                np.ones(len(candidates), dtype=np.float64) / len(candidates)
                if candidates[0].kind == "self_play"
                else pfsp_probabilities(candidates, self.pfsp_alpha, self.uniform_mix)
            )
            entries.extend(candidates)
            weights.extend(source_weight / total * within)
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        return entries, probabilities

    def sample_pair(self, rng, current_model_available=False):
        entries, probabilities = self._flattened_distribution(current_model_available)
        if not self.allow_duplicates and len(entries) < 2:
            raise ValueError("Two distinct opponent entries are required")
        first = int(rng.choice(len(entries), p=probabilities))
        if self.allow_duplicates:
            second = int(rng.choice(len(entries), p=probabilities))
        else:
            remaining = probabilities.copy()
            remaining[first] = 0.0
            remaining /= remaining.sum()
            second = int(rng.choice(len(entries), p=remaining))
        selected = entries[first], entries[second]
        for entry in selected:
            if entry.kind != "self_play":
                entry.selected += 1
        return selected

    def add_snapshot(self, step, path, metrics):
        checkpoint = Path(path)
        try:
            checkpoint_path = str(checkpoint.relative_to(self.model_root))
        except ValueError:
            checkpoint_path = str(checkpoint)
        entry = PoolEntry(
            id=f"snapshot-{int(step)}", kind="snapshot", name=f"RLBOASnapshot{int(step)}",
            checkpoint_path=checkpoint_path, added_step=int(step),
            snapshot_agent_utility=float(metrics["agent_utility"]),
            snapshot_score=float(metrics["negotiation_score"]),
        )
        if any(existing.id == entry.id for existing in self.entries):
            raise ValueError(f"Duplicate pool entry: {entry.id}")
        self.entries.append(entry)
        return entry

    def prune(self, protected_ids=()):
        protected = set(protected_ids)
        removed = []
        while len(self.entries) > self.max_size:
            candidates = [
                entry for entry in self.entries
                if entry.kind == "snapshot" and entry.id not in protected
            ]
            if not candidates:
                raise ValueError("Pool cannot prune scripted or protected entries")
            victim = min(candidates, key=snapshot_pruning_key)
            self.entries.remove(victim)
            removed.append(victim.id)
        return removed

    def to_dict(self):
        return {
            "format_version": POOL_FORMAT_VERSION,
            "max_size": self.max_size,
            "pfsp_alpha": self.pfsp_alpha,
            "uniform_mix": self.uniform_mix,
            "allow_duplicates": self.allow_duplicates,
            "source_probabilities": self.source_probabilities,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path=None):
        destination = Path(path) if path else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path, model_root=None):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format_version") != POOL_FORMAT_VERSION:
            raise ValueError("Unsupported pool format")
        source = data["source_probabilities"]
        pool = cls(
            path.parent, [], max_size=data["max_size"], pfsp_alpha=data["pfsp_alpha"],
            uniform_mix=data["uniform_mix"], allow_duplicates=data["allow_duplicates"],
            self_play_probability=source["self_play"],
            scripted_probability=source["scripted"],
            snapshot_probability=source["snapshot"], path=path, model_root=model_root,
        )
        pool.entries = [PoolEntry.from_dict(entry) for entry in data["entries"]]
        for entry in pool.entries:
            if entry.kind == "snapshot" and not (pool.model_root / entry.checkpoint_path).is_file():
                raise FileNotFoundError(f"Missing snapshot: {entry.checkpoint_path}")
        return pool

