"""RAMDocs oracle-claim fusion benchmark for DSm-RAG.

The document-level `answer` field is used as the output of a perfect claim
extractor. The `type`, `gold_answers`, and `wrong_answers` fields are never used
by a fusion method; they are used only to construct declared stress conditions
and to evaluate predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_experiment import dempster_all, ece, pcr6_all, pignistic


STOP = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "is", "was", "what", "who", "when", "where"}
METHODS = (
    "majority_topk",
    "relevance_topk",
    "dst_global_top1",
    "dst_global_topk",
    "dst_factorized",
    "dst_factorized_provenance",
    "dsmt_pcr6_factorized",
    "dsmt_pcr6_provenance",
)
VARIANTS = ("original", "duplicate_misinfo_x3", "misinfo_dominant")


def norm(text: str) -> str:
    text = text.lower().replace("�", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> set[str]:
    return {x for x in norm(text).split() if len(x) > 2 and x not in STOP}


def answer_key(text: str) -> str:
    return norm(str(text))


def document_confidence(question: str, entity: str, document: str) -> float:
    q = tokens(question)
    e = tokens(entity)
    d = tokens(document)
    q_overlap = len(q & d) / max(1, len(q))
    e_overlap = len(e & d) / max(1, len(e))
    phrase = float(norm(entity) in norm(document))
    value = 0.50 + 0.16 * q_overlap + 0.18 * e_overlap + 0.08 * phrase
    return float(np.clip(value, 0.50, 0.90))


def assign_entity(document: str, entities: list[str]) -> tuple[int, float]:
    dnorm = norm(document)
    dtok = tokens(document)
    scores = []
    for entity in entities:
        etok = tokens(entity)
        overlap = len(etok & dtok) / max(1, len(etok))
        phrase = float(norm(entity) in dnorm)
        parenthetical = re.findall(r"\(([^)]+)\)", entity)
        qualifier = float(any(norm(x) in dnorm for x in parenthetical))
        scores.append(overlap + 1.5 * phrase + 0.75 * qualifier)
    best = int(np.argmax(scores))
    return best, float(scores[best])


def provenance_id(text: str) -> str:
    cleaned = norm(text)
    return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16]


def variant_documents(item: dict, variant: str) -> list[dict]:
    docs = [dict(d) for d in item["documents"]]
    for d in docs:
        d["provenance"] = provenance_id(d["text"])
    if variant == "original":
        return docs
    if variant == "duplicate_misinfo_x3":
        extras = []
        for d in docs:
            if d["type"] == "misinfo":
                extras.extend([dict(d), dict(d), dict(d)])
        return docs + extras
    if variant == "misinfo_dominant":
        correct = []
        seen_answers = set()
        for d in docs:
            key = answer_key(d["answer"])
            if d["type"] == "correct" and key not in seen_answers:
                correct.append(d)
                seen_answers.add(key)
        other = [d for d in docs if d["type"] != "correct"]
        mis = [d for d in other if d["type"] == "misinfo"]
        return correct + other + [dict(d) for d in mis for _ in range(2)]
    raise ValueError(variant)


def prepare_sources(item: dict, docs: list[dict]) -> tuple[list[str], list[dict]]:
    entities = list(item["disambig_entity"])
    candidates = []
    sources = []
    for d in docs:
        key = answer_key(d.get("answer", "unknown"))
        if not key or key == "unknown":
            continue
        if key not in candidates:
            candidates.append(key)
        entity_idx, entity_score = assign_entity(d["text"], entities)
        confidence = document_confidence(item["question"], entities[entity_idx], d["text"])
        sources.append({
            "answer": key,
            "entity": entity_idx,
            "entity_score": entity_score,
            "confidence": confidence,
            "provenance": d["provenance"],
        })
    return candidates, sources


def adjusted_confidences(sources: list[dict]) -> list[float]:
    counts = Counter(s["provenance"] for s in sources)
    out = []
    for s in sources:
        r = s["confidence"]
        g = counts[s["provenance"]]
        out.append(1.0 - (1.0 - r) ** (1.0 / g))
    return out


def combine_group(sources: list[dict], answers: list[str], rule: str, provenance: bool) -> tuple[np.ndarray, float]:
    if not answers:
        return np.array([]), 0.0
    theta = frozenset(answers)
    confs = adjusted_confidences(sources) if provenance else [s["confidence"] for s in sources]
    masses = []
    for source, r in zip(sources, confs):
        masses.append({frozenset((source["answer"],)): r, theta: 1.0 - r})
    if not masses:
        return np.ones(len(answers)) / len(answers), 0.0
    combined = dempster_all(masses, theta) if rule == "dempster" else pcr6_all(masses)
    probs = pignistic(combined, tuple(answers))
    return probs, float(probs.max())


def top_k(scores: dict[str, float], k: int) -> list[str]:
    return [x[0] for x in sorted(scores.items(), key=lambda z: (-z[1], z[0]))[:k]]


def predict(item: dict, docs: list[dict], method: str) -> tuple[list[str], float]:
    candidates, sources = prepare_sources(item, docs)
    k = len(item["disambig_entity"])
    if not candidates:
        return [], 0.0

    if method in ("majority_topk", "relevance_topk"):
        scores: dict[str, float] = defaultdict(float)
        for s in sources:
            scores[s["answer"]] += 1.0 if method == "majority_topk" else s["confidence"]
        selected = top_k(scores, min(k, len(scores)))
        total = sum(scores.values())
        confidence = sum(scores[x] for x in selected) / total if total else 0.0
        return selected, confidence

    if method in ("dst_global_top1", "dst_global_topk"):
        probs, _ = combine_group(sources, candidates, "dempster", False)
        scores = dict(zip(candidates, probs))
        count = 1 if method == "dst_global_top1" else min(k, len(candidates))
        selected = top_k(scores, count)
        return selected, float(sum(scores[x] for x in selected))

    rule = "pcr6" if method.startswith("dsmt") else "dempster"
    provenance = method.endswith("provenance")
    selected = []
    confidences = []
    unused = set(candidates)
    for entity_idx in range(k):
        group_sources = [s for s in sources if s["entity"] == entity_idx]
        group_answers = sorted({s["answer"] for s in group_sources})
        if not group_answers:
            continue
        probs, conf = combine_group(group_sources, group_answers, rule, provenance)
        choice = group_answers[int(np.argmax(probs))]
        if choice not in selected:
            selected.append(choice)
            unused.discard(choice)
            confidences.append(conf)
    if len(selected) < k and unused:
        global_probs, _ = combine_group(sources, candidates, rule, provenance)
        scores = dict(zip(candidates, global_probs))
        for candidate in top_k({x: scores[x] for x in unused}, k - len(selected)):
            selected.append(candidate)
            confidences.append(scores[candidate])
    confidence = float(math.prod(confidences) ** (1.0 / len(confidences))) if confidences else 0.0
    return selected, confidence


def set_metrics(pred: list[str], gold: list[str]) -> tuple[float, float, float, float]:
    p, g = set(pred), set(gold)
    tp = len(p & g)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(p == g), precision, recall, f1


def bootstrap_pair(rows: list[dict], target: str, comparator: str, variant: str, reps: int, seed: int) -> dict:
    tr = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == target}
    cr = {r["item_id"]: r for r in rows if r["variant"] == variant and r["method"] == comparator}
    ids = sorted(set(tr) & set(cr))
    diff = np.array([tr[i]["set_f1"] - cr[i]["set_f1"] for i in ids])
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return {
        "variant": variant,
        "target": target,
        "comparator": comparator,
        "f1_difference": float(diff.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "ramdocs_results")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in args.data.open(encoding="utf-8")]

    rows = []
    for item_id, item in enumerate(items):
        gold = [answer_key(x) for x in item["gold_answers"]]
        for variant in VARIANTS:
            docs = variant_documents(item, variant)
            counts = Counter(d["type"] for d in docs)
            for method in METHODS:
                pred, confidence = predict(item, docs, method)
                exact, precision, recall, f1 = set_metrics(pred, gold)
                rows.append({
                    "item_id": item_id,
                    "variant": variant,
                    "method": method,
                    "gold_count": len(gold),
                    "correct_docs": counts["correct"],
                    "misinfo_docs": counts["misinfo"],
                    "noise_docs": counts["noise"],
                    "misinfo_dominant": int(counts["misinfo"] >= counts["correct"] and counts["misinfo"] > 0),
                    "exact_set": exact,
                    "set_precision": precision,
                    "set_recall": recall,
                    "set_f1": f1,
                    "confidence": confidence,
                    "correct": exact,
                    "prediction": " || ".join(pred),
                    "gold": " || ".join(gold),
                })

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["method"])].append(row)
    summary = []
    for (variant, method), block in groups.items():
        exact = np.array([x["exact_set"] for x in block])
        conf = np.array([x["confidence"] for x in block])
        summary.append({
            "variant": variant,
            "method": method,
            "n": len(block),
            "exact_set": float(exact.mean()),
            "set_f1": float(np.mean([x["set_f1"] for x in block])),
            "precision": float(np.mean([x["set_precision"] for x in block])),
            "recall": float(np.mean([x["set_recall"] for x in block])),
            "ece_exact": ece(conf, exact),
            "mean_confidence": float(conf.mean()),
        })
    summary.sort(key=lambda x: (VARIANTS.index(x["variant"]), METHODS.index(x["method"])))

    comparisons = []
    contrasts = (
        ("dsmt_pcr6_factorized", "dst_factorized"),
        ("dsmt_pcr6_provenance", "dst_factorized_provenance"),
        ("dsmt_pcr6_provenance", "dsmt_pcr6_factorized"),
        ("dsmt_pcr6_provenance", "majority_topk"),
    )
    for variant in VARIANTS:
        for target, comparator in contrasts:
            comparisons.append(bootstrap_pair(rows, target, comparator, variant, args.bootstrap, 20260914))

    write_csv(args.output / "ramdocs_item_results.csv", rows)
    write_csv(args.output / "ramdocs_summary.csv", summary)
    write_csv(args.output / "ramdocs_bootstrap.csv", comparisons)
    print(json.dumps({"items": len(items), "predictions": len(rows), "summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
