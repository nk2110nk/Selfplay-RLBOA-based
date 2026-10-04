from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.env import RLBOAEnv
from selfplay.checkpoint import load_training_state
from selfplay.context import SelfPlayContext
from selfplay.pool import OpponentPool
from train import parse_args, train


class CountingRLBOAEnv(RLBOAEnv):
    def __init__(self, **kwargs):
        self.learner_decisions = 0
        super().__init__(**kwargs)

    def step(self, action):
        self.learner_decisions += 1
        return super().step(action)


def test_one_ppo_buffer_item_per_learner_decision():
    environment = CountingRLBOAEnv(
        domain="Laptop", opponent=["Boulware", "Conceder"],
        add_noise=False, seed=21,
    )
    vector = DummyVecEnv([lambda: environment])
    model = PPO("MlpPolicy", vector, n_steps=2, batch_size=2, n_epochs=1,
                seed=21, device="cpu")
    model.learn(total_timesteps=2)
    assert environment.learner_decisions == 2
    assert model.rollout_buffer.full
    assert model.rollout_buffer.buffer_size == 2
    vector.close()


def test_dummy_vec_env_workers_share_one_pool(tmp_path):
    pool = OpponentPool(
        tmp_path / "pool", ["Boulware", "Linear"],
        scripted_probability=1, snapshot_probability=0,
        model_root=tmp_path,
    )
    context = SelfPlayContext(pool, tmp_path, seed=31)
    vector = DummyVecEnv([
        lambda rank=rank: RLBOAEnv(
            domain="Laptop", opponent=["Boulware", "Linear"],
            selfplay_context=context, model_dir=tmp_path,
            add_noise=False, seed=31 + rank,
        )
        for rank in range(2)
    ])
    vector.reset()
    assert all(environment.selfplay_context is context for environment in vector.envs)
    assert all(environment.selfplay_context.pool is pool for environment in vector.envs)
    assert sum(entry.selected for entry in pool.entries) == 4
    vector.close()


def test_general_model_smoke(tmp_path):
    model_dir = train(parse_args([
        "--issue", "Laptop", "Car", "--agents", "Boulware", "Linear",
        "--model-type", "general", "--save-path", str(tmp_path),
        "--seed", "22", "--total-timesteps", "2", "--n-envs", "1",
        "--n-steps", "2", "--batch-size", "2", "--n-epochs", "1",
        "--snapshot-freq", "100", "--pool-eval-freq", "100",
        "--no-opponent-noise",
    ]))
    state = load_training_state(model_dir)
    assert state["global_step"] == 2
    assert state["config"]["model_type"] == "general"
    assert state["domains"] == ["Laptop", "Car"]


def test_periodic_evaluation_and_snapshot_exact_step(tmp_path):
    model_dir = train(parse_args([
        "--issue", "Laptop", "--agents", "Boulware",
        "--save-path", str(tmp_path), "--seed", "23",
        "--total-timesteps", "2", "--n-envs", "1", "--n-steps", "2",
        "--batch-size", "2", "--n-epochs", "1",
        "--snapshot-freq", "2", "--pool-eval-freq", "2",
        "--pool-eval-episodes", "1", "--scripted-probability", "1",
        "--snapshot-probability", "0", "--no-opponent-noise",
    ]))
    assert (model_dir / "evaluation" / "step-2.tsv").is_file()
    assert (model_dir / "pool" / "snapshots" / "snapshot-2.zip").is_file()
    state = load_training_state(model_dir)
    assert state["global_step"] == 2
    assert state["schedule"]["next_evaluation"] == 4
