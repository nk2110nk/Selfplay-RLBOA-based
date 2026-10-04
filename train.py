"""Train three-party PPO RLBOA with persistent PFSP self play."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import negmas
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.env import RLBOAEnv
from envs.compat import gym
from selfplay.callback import SelfPlayCallback
from selfplay.checkpoint import (
    load_training_state, restore_environment, restore_random_states,
)
from selfplay.context import SelfPlayContext
from selfplay.pool import OpponentPool


ISSUE_NAMES = ("Laptop", "ItexvsCypress", "IS_BT_Acquisition", "Grocery",
               "thompson", "Car", "EnergySmall_A")
AGENT_LIST = ("Boulware", "Linear", "Conceder", "Atlas3", "CUHKAgent")
CASES = ("case1", "case2", "case3")


def runtime_versions():
    return {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__,
        "negmas": negmas.__version__, "gym": getattr(gym, "__version__", "unknown"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", "-a", nargs="+", choices=AGENT_LIST, required=False)
    parser.add_argument("--issue", "--issues", "-i", nargs="+", choices=ISSUE_NAMES)
    parser.add_argument("--model-type", choices=("expert", "general"), default="expert")
    parser.add_argument("--save-path", "--save_path", "-sp", default="results")
    parser.add_argument("--resume", nargs="?", const=".")
    parser.add_argument("--total-timesteps", "--timesteps", "-t", type=int, default=100_000)
    parser.add_argument("--n-envs", "--num-envs", "-n", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", choices=CASES, default="case1")
    parser.add_argument("--n-actions", type=int, default=10)
    parser.add_argument("--pool-path", default="pool/pool.json")
    parser.add_argument("--max-pool-size", type=int, default=16)
    parser.add_argument("--pfsp-alpha", type=float, default=2.0)
    parser.add_argument("--uniform-mix", type=float, default=0.1)
    parser.add_argument("--allow-duplicate-opponents", dest="allow_duplicate_opponents",
                        action="store_true", default=True)
    parser.add_argument("--no-allow-duplicate-opponents", dest="allow_duplicate_opponents",
                        action="store_false")
    parser.add_argument("--self-play-probability", type=float, default=0.0)
    parser.add_argument("--scripted-probability", type=float, default=0.5)
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--snapshot-freq", type=int, default=10_000)
    parser.add_argument("--pool-eval-freq", type=int, default=10_000)
    parser.add_argument("--pool-eval-episodes", type=int, default=4)
    parser.add_argument("--dominance-tolerance", type=float, default=0.01)
    parser.add_argument("--min-agreement-rate", type=float, default=0.0)
    parser.add_argument("--score-length-weight", type=float, default=-0.005)
    parser.add_argument("--opponent-noise", dest="opponent_noise", action="store_true", default=True)
    parser.add_argument("--no-opponent-noise", dest="opponent_noise", action="store_false")
    args = parser.parse_args(argv)
    if not args.resume and (not args.issue or not args.agents):
        parser.error("--issue and --agents are required for a new run")
    if args.issue and args.model_type == "expert" and len(args.issue) != 1:
        parser.error("expert mode requires exactly one domain")
    if args.agents and args.model_type == "expert" and len(args.agents) not in (1, 2):
        parser.error("expert mode requires one duplicated agent or exactly two agents")
    positive = (args.total_timesteps, args.n_envs, args.n_steps, args.batch_size,
                args.n_epochs, args.snapshot_freq, args.pool_eval_freq,
                args.pool_eval_episodes, args.n_actions)
    if any(value < 1 for value in positive):
        parser.error("timesteps, PPO sizes and frequencies must be positive")
    if not args.resume:
        probabilities = (args.self_play_probability, args.scripted_probability,
                         args.snapshot_probability)
        if (any(not math.isfinite(value) or value < 0 for value in probabilities)
                or sum(probabilities) <= 0):
            parser.error("source probabilities must be finite, non-negative and not all zero")
        if not 0 <= args.uniform_mix <= 1 or args.pfsp_alpha < 0:
            parser.error("invalid PFSP parameters")
        if args.agents and args.max_pool_size < len(set(args.agents)) + 1:
            parser.error("pool must have room for scripted entries and one snapshot")
        if (args.agents and not args.allow_duplicate_opponents and
                len(set(args.agents)) < 2 and args.self_play_probability <= 0):
            parser.error("duplicates disabled requires two initially available entries")
        rollout_size = args.n_envs * args.n_steps
        if args.snapshot_freq % rollout_size or args.pool_eval_freq % rollout_size:
            parser.error("snapshot/evaluation frequencies must be divisible by n_envs*n_steps")
    return args


def model_directory(args, issues, agents):
    if args.resume:
        path = Path(args.resume).resolve()
        return path if path.is_dir() else path.parent
    pair = f"{agents[0]}-{agents[1]}" if len(agents) == 2 else "-".join(agents)
    domain = issues[0] if args.model_type == "expert" else "general"
    return (Path(args.save_path) / f"seed-{args.seed}" / args.case / "models" /
            args.model_type / pair / domain / "RLBOASelfPlay_Negotiator")


def new_config(args):
    agents = list(args.agents)
    if len(agents) == 1:
        agents *= 2
    return {
        "issues": list(args.issue), "agents": agents,
        "model_type": args.model_type, "seed": args.seed, "device": args.device,
        "case": args.case, "n_actions": args.n_actions, "n_envs": args.n_envs,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
        "gamma": args.gamma, "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range, "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef, "max_grad_norm": args.max_grad_norm,
        "pool_path": args.pool_path, "max_pool_size": args.max_pool_size,
        "pfsp_alpha": args.pfsp_alpha, "uniform_mix": args.uniform_mix,
        "allow_duplicate_opponents": args.allow_duplicate_opponents,
        "self_play_probability": args.self_play_probability,
        "scripted_probability": args.scripted_probability,
        "snapshot_probability": args.snapshot_probability,
        "snapshot_freq": args.snapshot_freq, "pool_eval_freq": args.pool_eval_freq,
        "pool_eval_episodes": args.pool_eval_episodes,
        "dominance_tolerance": args.dominance_tolerance,
        "min_agreement_rate": args.min_agreement_rate,
        "score_length_weight": args.score_length_weight,
        "opponent_noise": args.opponent_noise,
        "versions": runtime_versions(),
    }


def _write_config(path, config):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary.replace(path)


def train(args):
    state = load_training_state(args.resume, args.device) if args.resume else None
    config = dict(state["config"]) if state else new_config(args)
    if state:
        for supplied, key in ((args.issue, "issues"), (args.agents, "agents")):
            if supplied and list(supplied) != config[key]:
                raise ValueError(f"Resume {key} differs from checkpoint")
        config["device"] = args.device
        if config.get("versions") and config["versions"] != runtime_versions():
            raise ValueError(
                f"Resume dependency versions differ: {runtime_versions()} != {config['versions']}"
            )
    rollout = int(config["n_envs"]) * int(config["n_steps"])
    if config["snapshot_freq"] % rollout or config["pool_eval_freq"] % rollout:
        raise ValueError("checkpoint event frequencies are not divisible by rollout size")
    if state and int(state["action_count"]) != int(config["n_actions"]):
        raise ValueError("training-state action metadata does not match config")
    if state and list(state["observation_shape"]) != [7]:
        raise ValueError("training-state observation metadata is incompatible")
    root = model_directory(args, config["issues"], config["agents"])
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("tensorboard", "pool/snapshots", "evaluation", "csv"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _write_config(root / "config.json", config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    pool_path = Path(config["pool_path"])
    if not pool_path.is_absolute():
        pool_path = root / pool_path
    pool_path = pool_path.resolve()
    try:
        pool_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("--pool-path must be inside the model directory") from error
    if state:
        pool = OpponentPool.load(pool_path, model_root=root)
        if pool.to_dict() != state["pool_metadata"]:
            raise ValueError("pool.json does not match training_state.pt")
    else:
        pool = OpponentPool(
            pool_path.parent, config["agents"], config["max_pool_size"],
            config["pfsp_alpha"], config["uniform_mix"],
            config["allow_duplicate_opponents"], config["self_play_probability"],
            config["scripted_probability"], config["snapshot_probability"],
            path=pool_path, model_root=root,
        )
        pool.save()
    context = SelfPlayContext(pool, root, config["seed"], config["device"])

    def make_env(rank):
        return lambda: RLBOAEnv(
            domain=config["issues"], opponent=config["agents"], case=config["case"],
            n_actions=config["n_actions"], add_noise=config["opponent_noise"],
            random_train=True, selfplay_context=context, model_dir=root,
            seed=config["seed"] + rank,
        )
    env = DummyVecEnv([make_env(rank) for rank in range(config["n_envs"])])
    if state:
        model = PPO.load(str(root / "checkpoint.zip"), env=env, device=config["device"])
        if int(model.num_timesteps) != int(state["global_step"]):
            raise ValueError("checkpoint.zip and training_state.pt timestep mismatch")
        if int(model.action_space.n) != int(config["n_actions"]):
            raise ValueError("checkpoint action space does not match config")
        if list(model.observation_space.shape) != [7]:
            raise ValueError("checkpoint observation space is incompatible")
        restore_random_states(state["random_states"], context)
        restore_environment(env, model, state.get("environment"), context, pool)
        schedule = state["schedule"]
    else:
        model = PPO(
            "MlpPolicy", env, seed=config["seed"], device=config["device"], verbose=1,
            n_steps=config["n_steps"], batch_size=config["batch_size"],
            n_epochs=config["n_epochs"], learning_rate=config["learning_rate"],
            gamma=config["gamma"], gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"], ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"], max_grad_norm=config["max_grad_norm"],
            tensorboard_log=str(root / "tensorboard"),
        )
        schedule = None
    context.current_model = model
    model.set_logger(configure(str(root / "tensorboard"), ["stdout", "tensorboard"]))
    current = int(model.num_timesteps)
    remaining = int(args.total_timesteps) - current
    if remaining < 0:
        raise ValueError(f"target {args.total_timesteps} is below checkpoint step {current}")
    if remaining and remaining % rollout:
        raise ValueError(f"remaining timesteps must be divisible by rollout size {rollout}")
    callback = SelfPlayCallback(config, root, pool, context, schedule)
    if remaining:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                    callback=callback, progress_bar=False)
    else:
        callback.model = model
        callback.num_timesteps = model.num_timesteps
        callback._save()
    env.close()
    print(f"saved_model:{root / 'checkpoint.zip'}")
    return root


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
