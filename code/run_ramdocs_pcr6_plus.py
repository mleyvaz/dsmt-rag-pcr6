"""External PCR6+ validation on frozen RAMDocs NLI scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from pcr6_plus import dempster_all, nonspecificity, pcr6_all, pcr6_plus_all, pignistic
from run_ramdocs_experiment import answer_key, provenance_id, set_metrics


RULES = {"dempster": dempster_all, "pcr6": pcr6_all, "pcr6_plus": pcr6_plus_all}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def source_bba(item_id: int, doc_index: int, candidates: list[str], score_map: dict):
    records = [score_map[(item_id, doc_index, candidate)] for candidate in candidates]
    support = np.asarray([row["entailment"] * (1.0 - row["contradiction"]) for row in records], dtype=float)
    order = sorted(range(len(candidates)), key=lambda j: (-support[j], candidates[j]))
    first, second = order[:2]
    row = records[first]
    commitment = float(np.clip(row["entailment"] * (1.0 - 0.5 * row["neutral"]) * (1.0 - 0.5 * row["contradiction"]), 0.05, 0.95))
    separation = float(np.clip((support[first] - support[second]) / max(support[first], 1e-12), 0.0, 1.0))
    theta = frozenset(candidates)
    mass = {
        frozenset((candidates[first],)): commitment * separation,
        frozenset((candidates[first], candidates[second])): commitment * (1.0 - separation),
        theta: 1.0 - commitment,
    }
    collapsed = {}
    for focal, value in mass.items():
        collapsed[focal] = collapsed.get(focal, 0.0) + value
    return {focal: value for focal, value in collapsed.items() if value > 1e-15}, commitment


def multiclass_brier(probs: list[float], candidates: list[str], gold: set[str]):
    present = [answer for answer in candidates if answer in gold]
    if not present:
        return math.nan
    target = np.asarray([1.0 / len(present) if answer in present else 0.0 for answer in candidates])
    return float(np.square(np.asarray(probs) - target).sum())


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(rows: list[dict], subset: str, metric: str, reps: int, seed: int):
    plus = {row["item_id"]: row for row in rows if row["rule"] == "pcr6_plus" and row[subset]}
    ordinary = {row["item_id"]: row for row in rows if row["rule"] == "pcr6" and row[subset]}
    ids = sorted(set(plus) & set(ordinary))
    differences = np.asarray([plus[i][metric] - ordinary[i][metric] for i in ids], dtype=float)
    differences = differences[np.isfinite(differences)]
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for start in range(0, reps, 1000):
        stop = min(start + 1000, reps)
        indexes = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        draws[start:stop] = differences[indexes].mean(axis=1)
    return {"subset": subset, "metric": metric, "contrast": "pcr6_plus_minus_pcr6", "n": len(differences), "difference": float(differences.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "ramdocs_pcr6_plus_results")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(args.data)
    nli_rows = load_jsonl(args.nli)
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
        for rule_name, rule in RULES.items():
            fused = rule(masses)
            probs = pignistic(fused, candidates)
            order = sorted(range(len(candidates)), key=lambda j: (-probs[j], candidates[j]))
            prediction = [candidates[j] for j in order[:k]]
            exact, precision, recall, f1 = set_metrics(prediction, list(gold))
            rows.append(
                {
                    "item_id": item_id,
                    "rule": rule_name,
                    "eligible": True,
                    "applicability": applicability,
                    "candidate_count": len(candidates),
                    "source_count": len(retained),
                    "gold_count": len(gold),
                    "exact_set": exact,
                    "set_precision": precision,
                    "set_recall": recall,
                    "set_f1": f1,
                    "brier": multiclass_brier(probs, candidates, gold),
                    "nonspecificity": nonspecificity(fused),
                    "prediction": " || ".join(prediction),
                    "gold": " || ".join(sorted(gold)),
                }
            )

    summaries = []
    for subset in ("eligible", "applicability"):
        for rule_name in RULES:
            block = [row for row in rows if row["rule"] == rule_name and row[subset]]
            summaries.append({"subset": subset, "rule": rule_name, "n": len(block), "exact_set": float(np.mean([row["exact_set"] for row in block])), "set_f1": float(np.mean([row["set_f1"] for row in block])), "brier": float(np.nanmean([row["brier"] for row in block])), "nonspecificity": float(np.mean([row["nonspecificity"] for row in block]))})
    comparisons = []
    for subset in ("eligible", "applicability"):
        for metric in ("brier", "exact_set", "set_f1"):
            comparisons.append(paired_bootstrap(rows, subset, metric, args.bootstrap, 20260916 + len(comparisons)))

    write_csv(args.output / "ramdocs_pcr6_plus_item_results.csv", rows)
    write_csv(args.output / "ramdocs_pcr6_plus_summary.csv", summaries)
    write_csv(args.output / "ramdocs_pcr6_plus_bootstrap.csv", comparisons)
    metadata = {"data": str(args.data), "nli": str(args.nli), "items_loaded": len(items), "nli_pairs": len(nli_rows), "maximum_sources": 6, "mapping": "protocol_v1.4"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

