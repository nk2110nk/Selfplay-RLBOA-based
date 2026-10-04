import json
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.env import RLBOAEnv
from selfplay.checkpoint import (
    capture_environment, load_training_state, restore_environment, save_checkpoint,
)
from selfplay.context import SelfPlayContext
from selfplay.entry import PoolEntry
from selfplay.pool import OpponentPool
from selfplay.snapshot_policy import FrozenPolicy, factory_for_entry, save_snapshot
from train import parse_args, train


def _model():
    env = RLBOAEnv(domain="Laptop", opponent=["Boulware", "Boulware"],
                   test=True, add_noise=False)
    return PPO("MlpPolicy", env, n_steps=2, batch_size=2, n_epochs=1,
               device="cpu", seed=1), env


def test_frozen_policy_has_no_gradients():
    model, env = _model()
    frozen = FrozenPolicy(model.policy)
    assert not frozen.policy.training
    assert all(not parameter.requires_grad for parameter in frozen.policy.parameters())
    env.close()


def test_snapshot_observers_are_independent(tmp_path):
    model, base_env = _model()
    config = {"issues": ["Laptop"], "case": "case1", "n_actions": 10}
    path = save_snapshot(tmp_path, model, config, 2,
                         {"agent_utility": .5, "negotiation_score": 1.0})
    entry = PoolEntry("snapshot-2", "snapshot", "snap",
                      checkpoint_path=str(path.relative_to(tmp_path)))
    factory = factory_for_entry(entry, tmp_path)
    env = RLBOAEnv(domain="Laptop", case="case1", test=True, add_noise=False,
                   fixed_entries=[entry, entry], fixed_factories=[factory, factory],
                   model_dir=tmp_path)
    env.reset()
    left, right = env.opponents
    assert left is not right
    assert left.observer is not right.observer
    assert left.om is not right.om
    done = False
    info = None
    while not done:
        result = env.step(9)
        if len(result) == 5:
            _, _, terminated, truncated, info = result
            done = terminated or truncated
        else:
            _, _, done, info = result
    assert info["opponent1_source"] == "snapshot"
    assert info["opponent2_source"] == "snapshot"
    env.close(); base_env.close()


def test_current_self_play_one_hundred_percent_smoke(tmp_path):
    model, base_env = _model()
    pool = OpponentPool(
        tmp_path / "pool", ["Boulware"], self_play_probability=1,
        scripted_probability=0, snapshot_probability=0, model_root=tmp_path,
    )
    context = SelfPlayContext(pool, tmp_path, seed=5)
    context.current_model = model
    env = RLBOAEnv(domain="Laptop", case="case1", test=True, add_noise=False,
                   selfplay_context=context, model_dir=tmp_path)
    env.reset()
    assert all(entry.kind == "self_play" for entry in env.current_entries)
    assert env.opponents[0].policy is not env.opponents[1].policy
    assert len(env.step(9)) in (4, 5)
    env.close(); base_env.close()


def test_atomic_checkpoint_round_trip(tmp_path):
    model, env = _model()
    pool = OpponentPool(tmp_path / "pool", ["Boulware"],
                        scripted_probability=1, snapshot_probability=0,
                        model_root=tmp_path)
    context = SelfPlayContext(pool, tmp_path, seed=7)
    config = {"issues": ["Laptop"], "agents": ["Boulware", "Boulware"],
              "case": "case1", "n_actions": 10}
    model.num_timesteps = 12
    save_checkpoint(tmp_path, model, pool, config, context,
                    {"next_evaluation": 20, "next_snapshot": 20})
    assert (tmp_path / "checkpoint.zip").is_file()
    assert not (tmp_path / "checkpoint.tmp.zip").exists()
    state = load_training_state(tmp_path)
    assert state["global_step"] == 12
    assert state["schedule"]["next_evaluation"] == 20
    assert load_training_state(tmp_path / "checkpoint.zip")["global_step"] == 12
    env.close()


def test_restored_environment_starts_at_fresh_episode_boundary(tmp_path):
    pool = OpponentPool(tmp_path / "pool", ["Boulware"],
                        scripted_probability=1, snapshot_probability=0,
                        model_root=tmp_path)
    context = SelfPlayContext(pool, tmp_path, seed=17)
    source = DummyVecEnv([lambda: RLBOAEnv(
        domain="Laptop", opponent=["Boulware", "Boulware"],
        selfplay_context=context, model_dir=tmp_path, add_noise=False, seed=17,
    )])
    model = PPO("MlpPolicy", source, n_steps=2, batch_size=2, n_epochs=1,
                device="cpu", seed=17)
    model._last_obs = source.reset()
    model._last_episode_starts = np.ones((1,), dtype=bool)
    saved = capture_environment(source, model)

    target = DummyVecEnv([lambda: RLBOAEnv(
        domain="Laptop", opponent=["Boulware", "Boulware"],
        selfplay_context=context, model_dir=tmp_path, add_noise=False, seed=17,
    )])
    restore_environment(target, model, saved, context, pool)
    restored = target.envs[0]
    assert restored.session is not None
    assert restored.state is None
    assert model._last_episode_starts.tolist() == [True]
    target.step(np.asarray([9]))
    source.close(); target.close()


def test_context_rng_restore(tmp_path):
    pool = OpponentPool(tmp_path, ["A", "B"], scripted_probability=1,
                        snapshot_probability=0)
    context = SelfPlayContext(pool, tmp_path, seed=91)
    state = context.state_dict()
    expected = context.rng.random()
    context.load_state_dict(state)
    assert context.rng.random() == expected


def test_resume_reaches_requested_target(tmp_path):
    common = [
        "--issue", "Laptop", "--agents", "Boulware", "Boulware",
        "--seed", "101", "--n-envs", "1", "--n-steps", "2",
        "--batch-size", "2", "--n-epochs", "1",
        "--snapshot-freq", "100", "--pool-eval-freq", "100",
        "--scripted-probability", "1", "--snapshot-probability", "0",
        "--no-opponent-noise",
    ]
    split = train(parse_args(common + ["--save-path", str(tmp_path / "split"),
                                       "--total-timesteps", "2"]))
    train(parse_args(["--resume", str(split), "--total-timesteps", "4"]))
    split_model = PPO.load(str(split / "checkpoint.zip"))
    assert split_model.num_timesteps == 4
    state = load_training_state(split)
    assert state["global_step"] == 4
    assert isinstance(state["environment"]["envs"][0], dict)
