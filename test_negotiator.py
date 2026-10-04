"""Evaluate a trained self-play RLBOA checkpoint against scripted agents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from stable_baselines3 import PPO

from envs.env import RLBOAEnv
from results import result_path, row_from_info, write_results
from selfplay.checkpoint import load_training_state
from selfplay.entry import PoolEntry


AGENTS = ("Boulware", "Linear", "Conceder", "Atlas3", "CUHKAgent")
DOMAINS = ("Laptop", "ItexvsCypress", "IS_BT_Acquisition", "Grocery",
           "thompson", "Car", "EnergySmall_A")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--load-path", "-m", required=True)
    parser.add_argument("--domain", "--issue", "-i", choices=DOMAINS, required=True)
    parser.add_argument("--opponent1", choices=AGENTS, required=True)
    parser.add_argument("--opponent2", choices=AGENTS, required=True)
    parser.add_argument("--case", choices=("case1", "case2", "case3"), default="case1")
    parser.add_argument("--model-type", choices=("expert", "general"), default="expert")
    parser.add_argument("--episodes", "-e", type=int, default=100)
    parser.add_argument("--n-actions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=False)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--noise", dest="noise", action="store_true", default=False)
    parser.add_argument("--no-noise", dest="noise", action="store_false")
    parser.add_argument("--export-root")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


def resolve_checkpoint(value):
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "checkpoint.zip"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluate(args):
    checkpoint = resolve_checkpoint(args.model)
    model_dir = checkpoint.parent
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.domain not in config["issues"]:
        raise ValueError(f"Checkpoint is not compatible with domain {args.domain}")
    if args.case != config["case"]:
        raise ValueError(f"Checkpoint case is {config['case']}, not {args.case}")
    if int(args.n_actions) != int(config["n_actions"]):
        raise ValueError("--n-actions does not match checkpoint config")
    if args.model_type != config["model_type"]:
        raise ValueError("--model-type does not match checkpoint config")
    model = PPO.load(str(checkpoint), device=args.device)
    if int(model.action_space.n) != int(args.n_actions):
        raise ValueError("Checkpoint action space is incompatible")
    if list(model.observation_space.shape) != [7]:
        raise ValueError("Checkpoint observation space is incompatible")
    training_state = load_training_state(model_dir, args.device)
    if int(training_state["global_step"]) != int(model.num_timesteps):
        raise ValueError("checkpoint.zip and training_state.pt timestep mismatch")
    names = (args.opponent1, args.opponent2)
    entries = tuple(PoolEntry(f"scripted-{name}", "scripted", name) for name in names)
    rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        env = RLBOAEnv(
            domain=args.domain, opponent=names, case=args.case, test=True,
            n_actions=args.n_actions, add_noise=args.noise, random_train=False,
            fixed_entries=entries, fixed_factories=names, seed=episode_seed,
            record_pool=False,
        )
        reset = env.reset(seed=episode_seed)
        observation = reset[0] if isinstance(reset, tuple) else reset
        done, info = False, None
        while not done:
            action, _ = model.predict(observation, deterministic=args.deterministic)
            result = env.step(action)
            if len(result) == 5:
                observation, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                observation, _, done, info = result
        rows.append(row_from_info(info, episode_seed,
                                  style="deterministic" if args.deterministic else "stochastic"))
        env.close()
    pair = f"{args.opponent1}-{args.opponent2}"
    output = write_results(
        result_path(model_dir, names, args.domain, args.deterministic, args.noise),
        rows,
    )
    manifest = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed, "case": args.case, "domain": args.domain,
        "opponents": list(names), "episodes": args.episodes,
        "deterministic": args.deterministic, "noise": args.noise,
        "n_actions": args.n_actions,
        "model_type": args.model_type,
        "training_step": int(model.num_timesteps),
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    if args.export_root:
        target = (
            Path(args.export_root) / f"seed-{args.seed}" / args.case / "evaluation" /
            args.model_type / pair / args.domain / args.case / output.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, target)
        shutil.copyfile(manifest_path, target.with_suffix(target.suffix + ".json"))
    print(f"saved_result:{output}")
    return output


def main(argv=None):
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
