"""SB3 callback coordinating evaluation, snapshots, logs and checkpoints."""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from results import write_results
from .checkpoint import save_checkpoint
from .evaluator import evaluate_pool
from .pool import is_dominant
from .snapshot_policy import save_snapshot


LOG_FIELDS = (
    "global_step", "event", "domain", "case", "agreement", "agent_utility",
    "length", "opponent1_id", "opponent2_id", "opponent1_source",
    "opponent2_source", "pool_size", "scripted_selected",
    "snapshot_selected", "self_play_selected",
)


class SelfPlayCallback(BaseCallback):
    def __init__(self, config, model_dir, pool, context, schedule=None, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.model_dir = Path(model_dir)
        self.pool = pool
        self.context = context
        schedule = schedule or {}
        self.next_evaluation = int(schedule.get("next_evaluation", config["pool_eval_freq"]))
        self.next_snapshot = int(schedule.get("next_snapshot", config["snapshot_freq"]))
        self.log_path = self.model_dir / "training_log.tsv"
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=LOG_FIELDS, delimiter="\t").writeheader()

    def schedule_state(self):
        return {"next_evaluation": self.next_evaluation,
                "next_snapshot": self.next_snapshot}

    def _append(self, event, info=None):
        info = info or {}
        selected = self.context.source_selected
        row = {
            "global_step": self.num_timesteps, "event": event,
            "domain": info.get("domain", ""), "case": info.get("case", ""),
            "agreement": info.get("agreement", ""),
            "agent_utility": info.get("agent_utility", ""),
            "length": info.get("length", ""),
            "opponent1_id": info.get("opponent1_id", ""),
            "opponent2_id": info.get("opponent2_id", ""),
            "opponent1_source": info.get("opponent1_source", ""),
            "opponent2_source": info.get("opponent2_source", ""),
            "pool_size": len(self.pool.entries),
            "scripted_selected": selected["scripted"],
            "snapshot_selected": selected["snapshot"],
            "self_play_selected": selected["self_play"],
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS, delimiter="\t").writerow(row)

    def _on_training_start(self):
        self.context.current_model = self.model

    def _evaluate(self):
        measurements, benchmark, rows = evaluate_pool(
            self.model, self.pool, self.config["issues"], [self.config["case"]],
            self.config["pool_eval_episodes"], self.model_dir, self.config["seed"],
            self.config["score_length_weight"], self.num_timesteps,
            self.config["device"], self.config["n_actions"], self.context.rng,
        )
        write_results(self.model_dir / "evaluation" / f"step-{self.num_timesteps}.tsv", rows)
        stats_path = self.model_dir / "pool" / "stats.tsv"
        write_header = not stats_path.exists()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ("step", "entry_id", "agreement_rate", "agent_utility",
                  "mean_length", "negotiation_score")
        with stats_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            if write_header:
                writer.writeheader()
            for entry_id, metrics in measurements.items():
                writer.writerow({"step": self.num_timesteps, "entry_id": entry_id, **metrics})
        self.pool.save()
        self._append("evaluation")
        return benchmark

    def _maybe_snapshot(self, benchmark):
        snapshots = [entry for entry in self.pool.entries if entry.kind == "snapshot"]
        incumbent = max(
            ({"agent_utility": entry.snapshot_agent_utility,
              "negotiation_score": entry.snapshot_score} for entry in snapshots),
            key=lambda value: (value["agent_utility"], value["negotiation_score"]),
            default={"agent_utility": float("-inf"), "negotiation_score": float("-inf")},
        )
        if not is_dominant(benchmark, incumbent, self.config["dominance_tolerance"],
                           self.config["min_agreement_rate"]):
            self._append("snapshot_rejected")
            return
        path = save_snapshot(self.model_dir, self.model, self.config,
                             self.num_timesteps, benchmark)
        added = self.pool.add_snapshot(self.num_timesteps, path, benchmark)
        self.pool.prune(protected_ids={added.id})
        self.pool.save()
        self._append("snapshot_added")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "agent_utility" in info:
                self._append("episode", info)
        return True

    def _process_events(self):
        benchmark = None
        if self.num_timesteps >= self.next_evaluation:
            if self.num_timesteps != self.next_evaluation:
                raise RuntimeError("Evaluation frequency must be divisible by n_envs")
            benchmark = self._evaluate()
            self.next_evaluation += self.config["pool_eval_freq"]
        if self.num_timesteps >= self.next_snapshot:
            if self.num_timesteps != self.next_snapshot:
                raise RuntimeError("Snapshot frequency must be divisible by n_envs")
            if benchmark is None:
                benchmark = self._evaluate()
            self._maybe_snapshot(benchmark)
            self.next_snapshot += self.config["snapshot_freq"]

    def _on_rollout_start(self):
        if self.num_timesteps:
            self._process_events()
            self._save()

    def _save(self):
        save_checkpoint(self.model_dir, self.model, self.pool, self.config,
                        self.context, self.schedule_state(), self.training_env)

    def _on_rollout_end(self):
        pass

    def _on_training_end(self):
        self._process_events()
        self._save()
