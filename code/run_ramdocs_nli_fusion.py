"""Fuse RAMDocs evidence using NLI-derived BBAs.

Candidate answers are enumerated from the dataset's document-level answer field,
but support, contradiction, confidence, and source selection come from NLI.
This is therefore an NLI-assisted fusion experiment, not yet free-form answer
generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_experiment import dempster_all, ece, pcr6_all, pignistic
from run_ramdocs_experiment import answer_key, assign_entity, provenance_id, set_metrics


METHODS = (
    "nli_pool_topk",
    "dst_nli_global_top1",
    "dst_nli_global_topk",
    "dst_nli_factorized_provenance",
    "dsmt_nli_pcr6_provenance",
)
VARIANTS = ("original", "duplicate_misinfo_x3", "misinfo_dominant")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def build_variant_docs(item: dict, variant: str) -> list[dict]:
    docs = []
    for idx, original in enumerate(item["documents"]):
        d = dict(original)
        d["base_index"] = idx
        d["provenance"] = provenance_id(d["text"])
        docs.append(d)
    if variant == "original":
        return docs
    if variant == "duplicate_misinfo_x3":
        return docs + [dict(d) for d in docs if d["type"] == "misinfo" for _ in range(3)]
    if variant == "misinfo_dominant":
        correct, seen = [], set()
        for d in docs:
            key = answer_key(d["answer"])
            if d["type"] == "correct" and key not in seen:
                correct.append(d)
                seen.add(key)
        other = [d for d in docs if d["type"] != "correct"]
        mis = [d for d in other if d["type"] == "misinfo"]
        return correct + other + [dict(d) for d in mis for _ in range(2)]
    raise ValueError(variant)


def collapse_provenance(docs: list[dict]) -> list[dict]:
    """Represent exact copies as one evidence source."""
    unique = {}
    for d in docs:
        unique.setdefault(d["provenance"], d)
    return list(unique.values())


def source_from_nli(item_id: int, doc: dict, candidates: list[str], score_map: dict) -> dict:
    records = [score_map[(item_id, doc["base_index"], candidate)] for candidate in candidates]
    support = np.array([r["entailment"] * (1.0 - r["contradiction"]) for r in records], dtype=float)
    top = int(np.argmax(support))
    chosen = candidates[top]
    entail = float(records[top]["entailment"])
    neutral = float(records[top]["neutral"])
    contradiction = float(records[top]["contradiction"])
    # Commitment rises with entailment and falls with neutral/contradiction.
    confidence = float(np.clip(entail * (1.0 - 0.5 * neutral) * (1.0 - 0.5 * contradiction), 0.05, 0.95))
    return {"answer": chosen, "confidence": confidence, "entailment": entail, "neutral": neutral, "contradiction": contradiction}


def combine_simple(sources: list[dict], answers: list[str], rule: str) -> tuple[np.ndarray, float]:
    if not answers:
        return np.array([]), 0.0
    theta = frozenset(answers)
    masses = [{frozenset((s["answer"],)): s["confidence"], theta: 1.0 - s["confidence"]} for s in sources]
    if not masses:
        return np.ones(len(answers)) / len(answers), 0.0
    mass = dempster_all(masses, theta) if rule == "dempster" else pcr6_all(masses)
    probs = pignistic(mass, tuple(answers))
    return probs, float(probs.max())


def select_top(scores: dict[str, float], k: int) -> list[str]:
    return [x for x, _ in sorted(scores.items(), key=lambda z: (-z[1], z[0]))[:k]]


def predict(item_id: int, item: dict, docs: list[dict], score_map: dict, method: str) -> tuple[list[str], float]:
    candidates = []
    for d in item["documents"]:
        key = answer_key(d.get("answer", "unknown"))
        if key and key != "unknown" and key not in candidates:
            candidates.append(key)
    if not candidates:
        return [], 0.0
    k = len(item["disambig_entity"])
    evidence_docs = collapse_provenance(docs) if method.endswith("provenance") else docs
    sources = []
    for d in evidence_docs:
        src = source_from_nli(item_id, d, candidates, score_map)
        entity, _ = assign_entity(d["text"], item["disambig_entity"])
        src["entity"] = entity
        sources.append(src)

    if method == "nli_pool_topk":
        scores = defaultdict(float)
        for s in sources:
            scores[s["answer"]] += s["confidence"]
        pred = select_top(scores, min(k, len(scores)))
        total = sum(scores.values())
        return pred, sum(scores[x] for x in pred) / total if total else 0.0

    if method in ("dst_nli_global_top1", "dst_nli_global_topk"):
        probs, _ = combine_simple(sources, candidates, "dempster")
        scores = dict(zip(candidates, probs))
        count = 1 if method.endswith("top1") else min(k, len(candidates))
        pred = select_top(scores, count)
        return pred, float(sum(scores[x] for x in pred))

    rule = "pcr6" if method.startswith("dsmt") else "dempster"
    selected, confidences = [], []
    for entity in range(k):
        group = [s for s in sources if s["entity"] == entity]
        answers = sorted({s["answer"] for s in group})
        if not answers:
            continue
        probs, conf = combine_simple(group, answers, rule)
        choice = answers[int(np.argmax(probs))]
        if choice not in selected:
            selected.append(choice)
            confidences.append(conf)
    if len(selected) < k:
        remaining = [x for x in candidates if x not in selected]
        probs, _ = combine_simple(sources, candidates, rule)
        scores = dict(zip(candidates, probs))
        for choice in select_top({x: scores[x] for x in remaining}, k - len(selected)):
            selected.append(choice)
            confidences.append(scores[choice])
    confidence = float(math.prod(confidences) ** (1.0 / len(confidences))) if confidences else 0.0
    return selected, confidence


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(rows: list[dict], variant: str, target: str, comparator: str, reps: int = 4000) -> dict:
    t = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == target}
    c = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == comparator}
    ids = sorted(t)
    diff = np.array([t[i]["set_f1"] - c[i]["set_f1"] for i in ids])
    rng = np.random.default_rng(20260915)
    draws = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return {"variant": variant, "target": target, "comparator": comparator, "f1_difference": float(diff.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975))}


def paired_bootstrap_brier(rows: list[dict], variant: str, target: str, comparator: str, reps: int = 4000) -> dict:
    t = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == target}
    c = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == comparator}
    ids = sorted(t)
    diff = np.array([t[i]["brier_exact"] - c[i]["brier_exact"] for i in ids])
    rng = np.random.default_rng(20260916)
    draws = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return {"variant": variant, "target": target, "comparator": comparator, "brier_difference": float(diff.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "interpretation": "negative favors target"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "ramdocs_nli_results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(args.data)
    nli_rows = load_jsonl(args.nli)
    score_map = {(r["item_id"], r["doc_index"], r["candidate"]): r for r in nli_rows}

    rows = []
    for item_id, item in enumerate(items):
        gold = [answer_key(x) for x in item["gold_answers"]]
        for variant in VARIANTS:
            docs = build_variant_docs(item, variant)
            for method in METHODS:
                pred, confidence = predict(item_id, item, docs, score_map, method)
                exact, precision, recall, f1 = set_metrics(pred, gold)
                rows.append({"item_id": item_id, "variant": variant, "method": method, "gold_count": len(gold), "exact_set": exact, "set_precision": precision, "set_recall": recall, "set_f1": f1, "confidence": confidence, "brier_exact": (confidence - exact) ** 2, "nll_exact": -math.log(max(1e-12, confidence if exact else 1.0 - confidence)), "prediction": " || ".join(pred), "gold": " || ".join(gold)})

    summary = []
    for variant in VARIANTS:
        for method in METHODS:
            block = [r for r in rows if r["variant"] == variant and r["method"] == method]
            correct = np.array([r["exact_set"] for r in block])
            conf = np.array([r["confidence"] for r in block])
            summary.append({"variant": variant, "method": method, "n": len(block), "exact_set": float(correct.mean()), "set_f1": float(np.mean([r["set_f1"] for r in block])), "precision": float(np.mean([r["set_precision"] for r in block])), "recall": float(np.mean([r["set_recall"] for r in block])), "brier_exact": float(np.mean([r["brier_exact"] for r in block])), "nll_exact": float(np.mean([r["nll_exact"] for r in block])), "ece_exact": ece(conf, correct), "mean_confidence": float(conf.mean())})

    comparisons = []
    brier_comparisons = []
    for variant in VARIANTS:
        comparisons.append(paired_bootstrap(rows, variant, "dsmt_nli_pcr6_provenance", "dst_nli_factorized_provenance"))
        comparisons.append(paired_bootstrap(rows, variant, "dsmt_nli_pcr6_provenance", "nli_pool_topk"))
        brier_comparisons.append(paired_bootstrap_brier(rows, variant, "dsmt_nli_pcr6_provenance", "dst_nli_factorized_provenance"))
        brier_comparisons.append(paired_bootstrap_brier(rows, variant, "dsmt_nli_pcr6_provenance", "nli_pool_topk"))
    write_csv(args.output / "nli_item_results.csv", rows)
    write_csv(args.output / "nli_summary.csv", summary)
    write_csv(args.output / "nli_bootstrap.csv", comparisons)
    write_csv(args.output / "nli_brier_bootstrap.csv", brier_comparisons)
    print(json.dumps({"items": len(items), "nli_pairs": len(nli_rows), "predictions": len(rows), "summary": summary, "comparisons": comparisons, "brier_comparisons": brier_comparisons}, indent=2))


if __name__ == "__main__":
    main()
