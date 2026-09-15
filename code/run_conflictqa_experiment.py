"""ConflictQA fusion benchmark for parametric versus retrieved evidence.

The experiment evaluates a calibrated RAG policy that treats parametric memory
and external evidence as conflicting sources. It includes copied-memory and
independent-memory stress tests to separate dependence from source count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from run_experiment import dempster_all, ece, pcr6_all, pignistic


METHODS = ("majority", "weighted_pool", "dst", "dst_provenance", "dsmt_pcr6", "dsmt_pcr6_provenance")
VARIANTS = ("balanced", "memory_repeat_x3", "independent_memory_x3", "external_repeat_x3")
ANSWERS = ("memory", "external")
THETA = frozenset(ANSWERS)


def source(candidate: str, confidence: float, provenance: str) -> dict:
    return {"candidate": candidate, "confidence": confidence, "provenance": provenance}


def evidence_confidence(answer: str, evidence: str) -> float:
    answer_tokens = {x for x in answer.lower().split() if len(x) > 3}
    evidence_lower = evidence.lower()
    coverage = sum(x in evidence_lower for x in answer_tokens) / max(1, len(answer_tokens))
    length_score = min(1.0, len(evidence.split()) / 120.0)
    return float(np.clip(0.72 + 0.10 * coverage + 0.05 * length_score, 0.72, 0.88))


def memory_confidence(popularity: float) -> float:
    return float(np.clip(0.53 + 0.045 * math.log10(max(1.0, popularity)), 0.53, 0.72))


def build_sources(item: dict, variant: str) -> list[dict]:
    mem_r = memory_confidence(float(item.get("popularity", 1)))
    ext_r = evidence_confidence(item["counter_answer"], item["counter_memory_aligned_evidence"])
    sources = [source("memory", mem_r, "parametric_model"), source("external", ext_r, "retrieved_document")]
    if variant == "memory_repeat_x3":
        sources.extend(source("memory", mem_r, "parametric_model") for _ in range(3))
    elif variant == "independent_memory_x3":
        sources.extend(source("memory", mem_r, f"independent_model_{i}") for i in range(3))
    elif variant == "external_repeat_x3":
        sources.extend(source("external", ext_r, "retrieved_document") for _ in range(3))
    elif variant != "balanced":
        raise ValueError(variant)
    return sources


def discount_confidences(sources: list[dict]) -> list[float]:
    counts = Counter(s["provenance"] for s in sources)
    return [1.0 - (1.0 - s["confidence"]) ** (1.0 / counts[s["provenance"]]) for s in sources]


def fuse(sources: list[dict], method: str) -> tuple[str, float]:
    if method == "majority":
        count = Counter(s["candidate"] for s in sources)
        pred = max(ANSWERS, key=lambda x: (count[x], x == "external"))
        return pred, count[pred] / len(sources)
    if method == "weighted_pool":
        scores = {x: 0.0 for x in ANSWERS}
        for s in sources:
            scores[s["candidate"]] += s["confidence"]
        total = sum(scores.values())
        pred = max(ANSWERS, key=lambda x: (scores[x], x == "external"))
        return pred, scores[pred] / total

    provenance = method.endswith("provenance")
    confs = discount_confidences(sources) if provenance else [s["confidence"] for s in sources]
    masses = [{frozenset((s["candidate"],)): r, THETA: 1.0 - r} for s, r in zip(sources, confs)]
    combined = pcr6_all(masses) if method.startswith("dsmt") else dempster_all(masses, THETA)
    probs = pignistic(combined, ANSWERS)
    idx = int(np.argmax(probs))
    return ANSWERS[idx], float(probs[idx])


def bootstrap(rows: list[dict], target: str, comparator: str, variant: str, reps: int = 2000) -> dict:
    t = np.array([r["correct"] for r in rows if r["variant"] == variant and r["method"] == target])
    c = np.array([r["correct"] for r in rows if r["variant"] == variant and r["method"] == comparator])
    diff = t - c
    rng = np.random.default_rng(20260915)
    draws = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return {"variant": variant, "target": target, "comparator": comparator, "accuracy_difference": float(diff.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975))}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "conflictqa_results")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.data.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.limit:
        items = items[: args.limit]

    rows = []
    for item_id, item in enumerate(items):
        for variant in VARIANTS:
            sources = build_sources(item, variant)
            for method in METHODS:
                pred, confidence = fuse(sources, method)
                rows.append({"item_id": item_id, "variant": variant, "method": method, "prediction": pred, "correct": int(pred == "external"), "confidence": confidence, "popularity": item.get("popularity", 0)})

    summary = []
    for variant in VARIANTS:
        for method in METHODS:
            block = [r for r in rows if r["variant"] == variant and r["method"] == method]
            correct = np.array([r["correct"] for r in block])
            confidence = np.array([r["confidence"] for r in block])
            summary.append({"variant": variant, "method": method, "n": len(block), "accuracy": float(correct.mean()), "ece": ece(confidence, correct), "mean_confidence": float(confidence.mean())})

    comparisons = []
    for variant in VARIANTS:
        comparisons.append(bootstrap(rows, "dsmt_pcr6", "dst", variant))
        comparisons.append(bootstrap(rows, "dsmt_pcr6_provenance", "dst_provenance", variant))
        comparisons.append(bootstrap(rows, "dsmt_pcr6_provenance", "majority", variant))
    write_csv(args.output / "conflictqa_item_results.csv", rows)
    write_csv(args.output / "conflictqa_summary.csv", summary)
    write_csv(args.output / "conflictqa_bootstrap.csv", comparisons)
    print(json.dumps({"items": len(items), "predictions": len(rows), "summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
