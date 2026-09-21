"""Frozen controlled experiment for multi-source PCR6+.

Run only after ``test_pcr6_plus.py`` passes.  The generator and endpoints are
defined in PCR6_PLUS_CONFIRMATORY_PROTOCOL_v1.4.md and the adjacent JSON
configuration.  The script writes item-level results, aggregate summaries,
and paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from pcr6_plus import TotalConflictError, dempster_all, nonspecificity, pcr6_all, pcr6_plus_all, pignistic


RULES = {"dempster": dempster_all, "pcr6": pcr6_all, "pcr6_plus": pcr6_plus_all}


def bba(singleton: str, specific: float, partial: tuple[str, ...], partial_mass: float, theta: frozenset[str]):
    if specific < 0 or partial_mass < 0 or specific + partial_mass > 1:
        raise ValueError("Invalid generated BBA")
    return {
        frozenset((singleton,)): specific,
        frozenset(partial): partial_mass,
        theta: 1.0 - specific - partial_mass,
    }


def generate_item(rng: np.random.Generator, atoms: tuple[str, ...]):
    truth = atoms[int(rng.integers(0, len(atoms)))]
    wrong = [atom for atom in atoms if atom != truth]
    rng.shuffle(wrong)
    w1, w2, w3 = wrong
    theta = frozenset(atoms)

    clean = [
        bba(truth, rng.uniform(0.58, 0.72), (truth, w1), rng.uniform(0.12, 0.20), theta),
        bba(truth, rng.uniform(0.54, 0.69), (truth, w2), rng.uniform(0.13, 0.22), theta),
        bba(truth, rng.uniform(0.48, 0.64), (truth, w3), rng.uniform(0.15, 0.24), theta),
        bba(truth, rng.uniform(0.42, 0.58), (truth, w1, w2), rng.uniform(0.16, 0.27), theta),
        bba(truth, rng.uniform(0.30, 0.46), (truth, w2, w3), rng.uniform(0.18, 0.30), theta),
    ]

    conflict = [
        bba(truth, rng.uniform(0.58, 0.76), (truth, w1), rng.uniform(0.10, 0.20), theta),
        bba(truth, rng.uniform(0.50, 0.69), (truth, w2), rng.uniform(0.12, 0.23), theta),
        bba(w1, rng.uniform(0.43, 0.61), (w1, w2), rng.uniform(0.17, 0.29), theta),
        bba(w2, rng.uniform(0.18, 0.31), (w1, w2), rng.uniform(0.43, 0.58), theta),
        bba(truth, rng.uniform(0.27, 0.42), (truth, w3), rng.uniform(0.20, 0.34), theta),
    ]
    vacuous = {theta: 1.0}
    near_vacuous = {frozenset((w3,)): 0.05, theta: 0.95}
    return truth, {
        "independent_clean": clean,
        "partial_ignorance_conflict": conflict,
        "vacuous_source_injection": conflict + [vacuous],
        "near_vacuous_source_injection": conflict + [near_vacuous],
    }


def multiclass_brier(probs: list[float], truth_index: int) -> float:
    y = np.zeros(len(probs), dtype=float)
    y[truth_index] = 1.0
    return float(np.square(np.asarray(probs) - y).sum())


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    value = 0.0
    for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        high = low + 1.0 / bins
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    conditions = sorted({row["condition"] for row in rows})
    for condition in conditions:
        for rule in RULES:
            block = [row for row in rows if row["condition"] == condition and row["rule"] == rule and not row["undefined"]]
            conf = np.asarray([row["confidence"] for row in block])
            correct = np.asarray([row["correct"] for row in block])
            output.append(
                {
                    "condition": condition,
                    "rule": rule,
                    "n": len(block),
                    "undefined": sum(row["undefined"] for row in rows if row["condition"] == condition and row["rule"] == rule),
                    "accuracy": float(correct.mean()) if len(block) else math.nan,
                    "brier": float(np.mean([row["brier"] for row in block])) if block else math.nan,
                    "nll": float(np.mean([row["nll"] for row in block])) if block else math.nan,
                    "ece": ece(conf, correct) if block else math.nan,
                    "nonspecificity": float(np.mean([row["nonspecificity"] for row in block])) if block else math.nan,
                }
            )
    return output


def paired_interval(rows: list[dict], condition: str, metric: str, reps: int, seed: int):
    plus = {row["item_id"]: row for row in rows if row["condition"] == condition and row["rule"] == "pcr6_plus"}
    ordinary = {row["item_id"]: row for row in rows if row["condition"] == condition and row["rule"] == "pcr6"}
    ids = sorted(set(plus) & set(ordinary))
    differences = np.asarray([plus[i][metric] - ordinary[i][metric] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for start in range(0, reps, 1000):
        stop = min(start + 1000, reps)
        indexes = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        draws[start:stop] = differences[indexes].mean(axis=1)
    return {
        "condition": condition,
        "contrast": "pcr6_plus_minus_pcr6",
        "metric": metric,
        "n": len(differences),
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pcr6_plus_confirmatory_config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "pcr6_plus_results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config["seed"])
    atoms = tuple(f"a{i + 1}" for i in range(config["frame_size"]))
    rows = []
    base_probabilities: dict[tuple[int, str], list[float]] = {}
    for item_id in range(config["n_items"]):
        truth, conditions = generate_item(rng, atoms)
        truth_index = atoms.index(truth)
        for condition in config["conditions"]:
            masses = conditions[condition]
            for rule_name, rule in RULES.items():
                try:
                    fused = rule(masses)
                except TotalConflictError:
                    rows.append({"item_id": item_id, "condition": condition, "rule": rule_name, "truth": truth, "prediction": "", "correct": 0, "confidence": math.nan, "brier": math.nan, "nll": math.nan, "nonspecificity": math.nan, "invariance_tv": math.nan, "undefined": 1})
                    continue
                probs = pignistic(fused, atoms)
                prediction_index = int(np.argmax(probs))
                if condition == "partial_ignorance_conflict":
                    base_probabilities[(item_id, rule_name)] = probs
                invariance_tv = math.nan
                if condition == "vacuous_source_injection":
                    base = base_probabilities[(item_id, rule_name)]
                    invariance_tv = 0.5 * float(np.abs(np.asarray(probs) - np.asarray(base)).sum())
                rows.append(
                    {
                        "item_id": item_id,
                        "condition": condition,
                        "rule": rule_name,
                        "truth": truth,
                        "prediction": atoms[prediction_index],
                        "correct": int(prediction_index == truth_index),
                        "confidence": probs[prediction_index],
                        "brier": multiclass_brier(probs, truth_index),
                        "nll": -math.log(max(1e-15, probs[truth_index])),
                        "nonspecificity": nonspecificity(fused),
                        "invariance_tv": invariance_tv,
                        "undefined": 0,
                    }
                )

    summary = summarize(rows)
    comparisons = []
    for condition in config["conditions"]:
        comparisons.append(paired_interval(rows, condition, "brier", config["bootstrap_replicates"], config["seed"] + len(comparisons) + 1))
    comparisons.append(paired_interval(rows, "vacuous_source_injection", "invariance_tv", config["bootstrap_replicates"], config["seed"] + 99))

    write_csv(args.output / "controlled_item_results.csv", rows)
    write_csv(args.output / "controlled_summary.csv", summary)
    write_csv(args.output / "controlled_bootstrap.csv", comparisons)
    metadata = {"config": config, "implementation": "pcr6_plus.py", "published_example_tests": "passed before run"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

