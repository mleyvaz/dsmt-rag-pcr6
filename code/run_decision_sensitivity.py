"""Decision-rule sensitivity for the PCR6+ confirmatory experiments.

This is a post-confirmatory robustness analysis requested during co-author
review.  It does not change the frozen fusion experiment.  It compares BetP,
DSmP(epsilon=0.001), maximum singleton belief, maximum singleton
plausibility, and minimum belief-interval distance on the same fused BBAs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from pcr6_plus import (
    belief_interval_distance_scores,
    dempster_all,
    dsmp,
    pcr6_all,
    pcr6_plus_all,
    pignistic,
    singleton_belief_scores,
    singleton_plausibility_scores,
)
from run_pcr6_plus_confirmatory import generate_item, multiclass_brier
from run_ramdocs_experiment import answer_key, provenance_id, set_metrics
from run_ramdocs_pcr6_plus import load_jsonl, source_bba


RULES = {"dempster": dempster_all, "pcr6": pcr6_all, "pcr6_plus": pcr6_plus_all}
DECISIONS = {
    "betp": pignistic,
    "dsmp_epsilon_0.001": dsmp,
    "max_belief": singleton_belief_scores,
    "max_plausibility": singleton_plausibility_scores,
    "min_belief_interval_distance": belief_interval_distance_scores,
}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordered(scores, candidates):
    return sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index]))


def controlled(config: dict):
    rng = np.random.default_rng(config["seed"])
    atoms = tuple(f"a{i + 1}" for i in range(config["frame_size"]))
    rows = []
    for item_id in range(config["n_items"]):
        truth, conditions = generate_item(rng, atoms)
        truth_index = atoms.index(truth)
        for condition in config["conditions"]:
            for rule_name, rule in RULES.items():
                fused = rule(conditions[condition])
                for decision_name, transform in DECISIONS.items():
                    scores = transform(fused, atoms)
                    prediction_index = ordered(scores, atoms)[0]
                    probabilistic = decision_name in {"betp", "dsmp_epsilon_0.001"}
                    rows.append(
                        {
                            "item_id": item_id,
                            "condition": condition,
                            "rule": rule_name,
                            "decision": decision_name,
                            "truth": truth,
                            "prediction": atoms[prediction_index],
                            "correct": int(prediction_index == truth_index),
                            "brier": multiclass_brier(scores, truth_index) if probabilistic else math.nan,
                        }
                    )
    summary = []
    for condition in config["conditions"]:
        for rule_name in RULES:
            for decision_name in DECISIONS:
                block = [r for r in rows if r["condition"] == condition and r["rule"] == rule_name and r["decision"] == decision_name]
                values = [r["brier"] for r in block if math.isfinite(r["brier"])]
                summary.append(
                    {
                        "condition": condition,
                        "rule": rule_name,
                        "decision": decision_name,
                        "n": len(block),
                        "accuracy": float(np.mean([r["correct"] for r in block])),
                        "brier": float(np.mean(values)) if values else math.nan,
                    }
                )
    return rows, summary


def ramdocs(data: Path, nli: Path):
    items = load_jsonl(data)
    nli_rows = load_jsonl(nli)
    score_map = {(row["item_id"], row["doc_index"], row["candidate"]): row for row in nli_rows}
    rows = []
    for item_id, item in enumerate(items):
        candidates = []
        for doc in item["documents"]:
            candidate = answer_key(doc.get("answer", "unknown"))
            if candidate and candidate != "unknown" and candidate not in candidates:
                candidates.append(candidate)
        if len(candidates) < 2:
            continue
        unique = {}
        for doc_index, doc in enumerate(item["documents"]):
            provenance = provenance_id(doc["text"])
            if provenance in unique:
                continue
            mass, commitment = source_bba(item_id, doc_index, candidates, score_map)
            unique[provenance] = (doc_index, mass, commitment)
        retained = sorted(unique.values(), key=lambda value: (-value[2], value[0]))[:6]
        if len(retained) < 2:
            continue
        masses = [entry[1] for entry in retained]
        theta = frozenset(candidates)
        has_partial = any(any(1 < len(focal) < len(theta) and value > 0 for focal, value in mass.items()) for mass in masses)
        applicability = len(retained) >= 3 and len(candidates) >= 3 and has_partial
        gold = {answer_key(answer) for answer in item["gold_answers"]}
        k = min(len(gold), len(candidates))
        present_gold = [candidate for candidate in candidates if candidate in gold]
        target = np.asarray([1.0 / len(present_gold) if candidate in present_gold else 0.0 for candidate in candidates])
        for rule_name, rule in RULES.items():
            fused = rule(masses)
            for decision_name, transform in DECISIONS.items():
                scores = transform(fused, candidates)
                prediction = [candidates[index] for index in ordered(scores, candidates)[:k]]
                exact, precision, recall, f1 = set_metrics(prediction, list(gold))
                probabilistic = decision_name in {"betp", "dsmp_epsilon_0.001"}
                rows.append(
                    {
                        "item_id": item_id,
                        "applicability": applicability,
                        "rule": rule_name,
                        "decision": decision_name,
                        "source_count": len(retained),
                        "candidate_count": len(candidates),
                        "exact_set": exact,
                        "set_f1": f1,
                        "brier": float(np.square(np.asarray(scores) - target).sum()) if probabilistic else math.nan,
                    }
                )
    summary = []
    for subset in ("eligible", "applicability"):
        for rule_name in RULES:
            for decision_name in DECISIONS:
                block = [r for r in rows if r["rule"] == rule_name and r["decision"] == decision_name and (subset == "eligible" or r["applicability"])]
                values = [r["brier"] for r in block if math.isfinite(r["brier"])]
                summary.append(
                    {
                        "subset": subset,
                        "rule": rule_name,
                        "decision": decision_name,
                        "n": len(block),
                        "exact_set": float(np.mean([r["exact_set"] for r in block])),
                        "set_f1": float(np.mean([r["set_f1"] for r in block])),
                        "brier": float(np.mean(values)) if values else math.nan,
                    }
                )
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pcr6_plus_confirmatory_config.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "decision_sensitivity_results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controlled_rows, controlled_summary = controlled(config)
    ramdocs_rows, ramdocs_summary = ramdocs(args.data, args.nli)
    write_csv(args.output / "controlled_decision_item_results.csv", controlled_rows)
    write_csv(args.output / "controlled_decision_summary.csv", controlled_summary)
    write_csv(args.output / "ramdocs_decision_item_results.csv", ramdocs_rows)
    write_csv(args.output / "ramdocs_decision_summary.csv", ramdocs_summary)
    metadata = {
        "analysis": "post-confirmatory decision-rule sensitivity",
        "dsmp_epsilon": 0.001,
        "decision_rules": list(DECISIONS),
        "fusion_rules": list(RULES),
        "seed": config["seed"],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"controlled": controlled_summary, "ramdocs": ramdocs_summary}, indent=2))


if __name__ == "__main__":
    main()
