import json

import numpy as np
import pytest

from selfplay.entry import PoolEntry
from selfplay.pfsp import pfsp_probabilities
from selfplay.pool import OpponentPool


def test_pfsp_probability_formula():
    entries = [PoolEntry("a", "scripted", "A", difficulty=0.2),
               PoolEntry("b", "scripted", "B", difficulty=0.8)]
    probabilities = pfsp_probabilities(entries, alpha=2.0, uniform_mix=0.1)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[1] > probabilities[0] > 0


def test_ordered_pair_and_duplicate_control(tmp_path):
    pool = OpponentPool(tmp_path, ["A", "B"], allow_duplicates=False,
                        scripted_probability=1, snapshot_probability=0)
    rng = np.random.default_rng(4)
    pairs = [tuple(entry.id for entry in pool.sample_pair(rng)) for _ in range(20)]
    assert all(left != right for left, right in pairs)
    assert ("scripted-A", "scripted-B") in pairs
    assert ("scripted-B", "scripted-A") in pairs


def test_same_name_duplicates_are_allowed(tmp_path):
    pool = OpponentPool(tmp_path, ["A"], allow_duplicates=True,
                        scripted_probability=1, snapshot_probability=0)
    pair = pool.sample_pair(np.random.default_rng(0))
    assert pair[0].id == pair[1].id == "scripted-A"


def test_pool_json_round_trip_and_pruning(tmp_path):
    root = tmp_path / "pool"
    pool = OpponentPool(root, ["A"], max_size=2,
                        scripted_probability=1, snapshot_probability=0,
                        model_root=tmp_path)
    for step in (10, 20):
        checkpoint = tmp_path / f"s{step}.zip"
        checkpoint.touch()
        pool.add_snapshot(step, checkpoint, {"agent_utility": step / 100,
                                             "negotiation_score": step})
    removed = pool.prune(protected_ids={"snapshot-20"})
    assert removed == ["snapshot-10"]
    path = pool.save()
    restored = OpponentPool.load(path, model_root=tmp_path)
    assert restored.to_dict() == pool.to_dict()
    assert json.loads(path.read_text())["format_version"] == 1
