"""Stable result schema shared by periodic and final evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from selfplay.pfsp import negotiation_score


FIELDS = (
    "my_util", "opp_util1", "opp_util2", "social", "nash", "agreement", "step",
    "style", "opponent1_id", "opponent2_id", "opponent1_source", "opponent2_source",
    "negotiation_score", "seed", "domain", "case",
)


def bool_tag(value):
    return "T" if value else "F"


def result_path(model_dir, agents, domain, deterministic=False, noise=False):
    return (
        Path(model_dir) / "csv" / f"{agents[0]}-{agents[1]}" / domain
        / f"det={deterministic}_noise={noise}"
        / f"{domain}-{agents[0]}-{agents[1]}-d{bool_tag(deterministic)}-n{bool_tag(noise)}.tsv"
    )


def row_from_info(info, seed, length_weight=-0.005, style="neutral"):
    my_util = float(info["agent_utility"])
    opp1 = float(info["opponent1_utility"])
    opp2 = float(info["opponent2_utility"])
    return {
        "my_util": my_util,
        "opp_util1": opp1,
        "opp_util2": opp2,
        "social": my_util + opp1 + opp2,
        "nash": my_util * opp1 * opp2,
        "agreement": info["state"]["agreement"],
        "step": info["length"],
        "style": style,
        "opponent1_id": info["opponent1_id"],
        "opponent2_id": info["opponent2_id"],
        "opponent1_source": info["opponent1_source"],
        "opponent2_source": info["opponent2_source"],
        "negotiation_score": negotiation_score(
            info["agreement"], my_util, info["length"], length_weight
        ),
        "seed": int(seed),
        "domain": info["domain"],
        "case": info["case"],
    }


def write_results(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination

