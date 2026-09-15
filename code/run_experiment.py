"""Controlled benchmark for conflict-aware DSm-RAG fusion.

The benchmark separates two questions:
1. Does a nonexclusive/refined frame help when claims overlap?
2. Does PCR6 handle partial conflict better than Dempster normalization?

It uses synthetic, fully auditable evidence packets. Results are evidence about
the implemented mechanisms, not about end-to-end LLM performance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATOMS = ("A_only", "B_only", "A_and_B", "C")
THETA = frozenset(ATOMS)
SEMANTIC = {
    "A": frozenset(("A_only", "A_and_B")),
    "B": frozenset(("B_only", "A_and_B")),
    "C": frozenset(("C",)),
    "THETA": THETA,
}
COARSE = {
    "A": frozenset(("A",)),
    "B": frozenset(("B",)),
    "C": frozenset(("C",)),
    "THETA": frozenset(("A", "B", "C")),
}
COARSE_TO_ATOMS = {
    frozenset(("A",)): SEMANTIC["A"],
    frozenset(("B",)): SEMANTIC["B"],
    frozenset(("C",)): SEMANTIC["C"],
    frozenset(("A", "B", "C")): THETA,
}


@dataclass(frozen=True)
class Source:
    masses: dict[frozenset[str], float]
    group: str


def normalize_mass(mass: dict[frozenset[str], float]) -> dict[frozenset[str], float]:
    total = sum(mass.values())
    if total <= 0:
        return {THETA: 1.0}
    return {k: v / total for k, v in mass.items() if v > 1e-15}


def simple_support(claim: str, confidence: float, group: str) -> Source:
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return Source({SEMANTIC[claim]: confidence, THETA: 1.0 - confidence}, group)


def categorical_support(weights: dict[str, float], group: str) -> Source:
    mass: dict[frozenset[str], float] = defaultdict(float)
    for claim, value in weights.items():
        mass[SEMANTIC[claim]] += value
    return Source(normalize_mass(dict(mass)), group)


def discount(source: Source, alpha: float) -> Source:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out: dict[frozenset[str], float] = defaultdict(float)
    for focal, value in source.masses.items():
        if focal == THETA:
            out[THETA] += alpha * value
        else:
            out[focal] += alpha * value
    out[THETA] += 1.0 - alpha
    return Source(normalize_mass(dict(out)), source.group)


def provenance_discount(sources: list[Source]) -> list[Source]:
    """Make repeated evidence from one provenance group worth one source in total.

    For simple support with confidence r, each of g duplicates receives
    r_eff = 1 - (1-r)^(1/g), so their combined ignorance equals that of one copy.
    For general BBAs, the same ratio is applied using non-ignorance as r.
    """
    counts: dict[str, int] = defaultdict(int)
    for source in sources:
        counts[source.group] += 1
    adjusted = []
    for source in sources:
        g = counts[source.group]
        if g == 1:
            adjusted.append(source)
            continue
        r = 1.0 - source.masses.get(THETA, 0.0)
        if r <= 1e-12:
            adjusted.append(source)
            continue
        r_eff = 1.0 - (1.0 - r) ** (1.0 / g)
        adjusted.append(discount(source, r_eff / r))
    return adjusted


def conjunctive(m1: dict, m2: dict) -> dict:
    out: dict[frozenset[str], float] = defaultdict(float)
    for x, mx in m1.items():
        for y, my in m2.items():
            out[x & y] += mx * my
    return dict(out)


def dempster_pair(m1: dict, m2: dict, theta: frozenset[str]) -> dict:
    out = conjunctive(m1, m2)
    conflict = out.pop(frozenset(), 0.0)
    if conflict >= 1.0 - 1e-12:
        return {theta: 1.0}
    return normalize_mass({k: v / (1.0 - conflict) for k, v in out.items()})


def dempster_all(masses: list[dict], theta: frozenset[str]) -> dict:
    out = {theta: 1.0}
    for mass in masses:
        out = dempster_pair(out, mass, theta)
    return out


def yager_all(masses: list[dict], theta: frozenset[str]) -> dict:
    out = {theta: 1.0}
    for mass in masses:
        combined = conjunctive(out, mass)
        combined[theta] = combined.get(theta, 0.0) + combined.pop(frozenset(), 0.0)
        out = normalize_mass(combined)
    return out


def pcr6_all(masses: list[dict]) -> dict:
    """Exact multi-source PCR6 for the focal combinations used here."""
    out: dict[frozenset[str], float] = defaultdict(float)
    items = [list(mass.items()) for mass in masses]
    for combo in itertools.product(*items):
        focals = [item[0] for item in combo]
        values = [item[1] for item in combo]
        product = math.prod(values)
        intersection = set(focals[0])
        for focal in focals[1:]:
            intersection.intersection_update(focal)
        if intersection:
            out[frozenset(intersection)] += product
        else:
            denom = sum(values)
            if denom > 0:
                for focal, value in zip(focals, values):
                    out[focal] += product * value / denom
    return normalize_mass(dict(out))


def pignistic(mass: dict[frozenset[str], float], atoms: tuple[str, ...]) -> np.ndarray:
    probs = {atom: 0.0 for atom in atoms}
    for focal, value in mass.items():
        if not focal:
            continue
        share = value / len(focal)
        for atom in focal:
            if atom in probs:
                probs[atom] += share
    arr = np.array([probs[atom] for atom in atoms], dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else np.ones(len(atoms)) / len(atoms)


def coarse_mass(source: Source) -> dict[frozenset[str], float]:
    out: dict[frozenset[str], float] = defaultdict(float)
    reverse = {value: COARSE[key] for key, value in SEMANTIC.items()}
    for focal, value in source.masses.items():
        out[reverse[focal]] += value
    return dict(out)


def coarse_to_atomic(mass: dict[frozenset[str], float]) -> np.ndarray:
    out = np.zeros(len(ATOMS))
    for coarse_focal, value in mass.items():
        mapped: set[str] = set()
        if coarse_focal == COARSE["THETA"]:
            mapped.update(THETA)
        else:
            for item in coarse_focal:
                mapped.update(SEMANTIC[item])
        if mapped:
            for atom in mapped:
                out[ATOMS.index(atom)] += value / len(mapped)
    return out / out.sum()


def fuse(sources: list[Source], method: str) -> np.ndarray:
    if method == "vote_pool":
        return np.mean([pignistic(s.masses, ATOMS) for s in sources], axis=0)
    if method == "dst_coarse_dempster":
        mass = dempster_all([coarse_mass(s) for s in sources], COARSE["THETA"])
        return coarse_to_atomic(mass)
    if method == "dst_refined_dempster":
        return pignistic(dempster_all([s.masses for s in sources], THETA), ATOMS)
    if method == "dst_refined_dempster_provenance":
        adjusted = provenance_discount(sources)
        return pignistic(dempster_all([s.masses for s in adjusted], THETA), ATOMS)
    if method == "dst_refined_yager":
        return pignistic(yager_all([s.masses for s in sources], THETA), ATOMS)
    if method == "dsmt_pcr6":
        return pignistic(pcr6_all([s.masses for s in sources]), ATOMS)
    if method == "dsmt_pcr6_provenance":
        adjusted = provenance_discount(sources)
        return pignistic(pcr6_all([s.masses for s in adjusted]), ATOMS)
    raise ValueError(method)


METHODS = (
    "vote_pool",
    "dst_coarse_dempster",
    "dst_refined_dempster",
    "dst_refined_dempster_provenance",
    "dst_refined_yager",
    "dsmt_pcr6",
    "dsmt_pcr6_provenance",
)


def wrong_claim(correct: str, rng: np.random.Generator) -> str:
    return str(rng.choice([x for x in ("A", "B", "C") if x != correct]))


def generate_packet(scenario: str, rng: np.random.Generator) -> tuple[str, list[Source]]:
    if scenario == "exclusive_low_conflict":
        truth = str(rng.choice(("A_only", "B_only", "C")))
        correct = {"A_only": "A", "B_only": "B", "C": "C"}[truth]
        sources = []
        for i in range(5):
            r = float(rng.uniform(0.72, 0.92))
            claim = correct if rng.random() < r else wrong_claim(correct, rng)
            sources.append(simple_support(claim, r, f"ind_{i}"))
        return truth, sources

    if scenario == "overlap_balanced":
        truth = "A_and_B"
        sources = []
        for i in range(6):
            u = rng.random()
            claim = "A" if u < 0.47 else "B" if u < 0.94 else "C"
            sources.append(simple_support(claim, float(rng.uniform(0.78, 0.94)), f"ind_{i}"))
        return truth, sources

    if scenario == "zadeh_overlap_trap":
        truth = "A_and_B"
        a = float(rng.uniform(0.93, 0.97))
        b = float(rng.uniform(0.90, 0.95))
        return truth, [
            categorical_support({"A": a, "C": 1.0 - a}, "ind_1"),
            categorical_support({"B": b, "C": 1.0 - b}, "ind_2"),
        ]

    if scenario == "exclusive_partial_conflict":
        truth = "A_only"
        return truth, [
            simple_support("A", float(rng.uniform(0.90, 0.98)), "ind_1"),
            simple_support("C", float(rng.uniform(0.80, 0.93)), "ind_2"),
            simple_support("A", float(rng.uniform(0.55, 0.78)), "ind_3"),
        ]

    if scenario == "duplicated_misinformation":
        truth = "A_only"
        sources = []
        for i in range(int(rng.integers(1, 4))):
            claim = "A" if rng.random() > 0.12 else "B"
            sources.append(simple_support(claim, float(rng.uniform(0.68, 0.94)), f"truth_{i}"))
        shared_r = float(rng.uniform(0.55, 0.84))
        for _ in range(int(rng.integers(3, 8))):
            sources.append(simple_support("C", shared_r, "copied_false_source"))
        return truth, sources

    if scenario == "independent_misinformation":
        truth = "A_only"
        sources = []
        for i in range(int(rng.integers(1, 4))):
            sources.append(simple_support("A", float(rng.uniform(0.68, 0.94)), f"truth_{i}"))
        for i in range(int(rng.integers(3, 8))):
            sources.append(simple_support("C", float(rng.uniform(0.55, 0.84)), f"false_{i}"))
        return truth, sources

    if scenario == "missing_evidence":
        truth = str(rng.choice(("A_only", "B_only", "A_and_B", "C")))
        valid = {
            "A_only": ("A",), "B_only": ("B",), "A_and_B": ("A", "B"), "C": ("C",)
        }[truth]
        sources = []
        for i in range(2):
            claim = str(rng.choice(valid))
            sources.append(simple_support(claim, float(rng.uniform(0.60, 0.80)), f"ind_{i}"))
        sources.append(simple_support(str(rng.choice(("A", "B", "C"))), 0.12, "weak_unknown"))
        return truth, sources

    raise ValueError(scenario)


SCENARIOS = (
    "exclusive_low_conflict",
    "overlap_balanced",
    "zadeh_overlap_trap",
    "exclusive_partial_conflict",
    "duplicated_misinformation",
    "independent_misinformation",
    "missing_evidence",
)


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi if hi < 1 else confidence <= hi)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)
    output = []
    for (scenario, method), items in grouped.items():
        probs = np.array([[x[f"p_{a}"] for a in ATOMS] for x in items])
        truth = np.array([ATOMS.index(x["truth"]) for x in items])
        pred = probs.argmax(axis=1)
        correct = pred == truth
        one_hot = np.eye(len(ATOMS))[truth]
        chosen = probs[np.arange(len(probs)), truth]
        conf = probs.max(axis=1)
        output.append({
            "scenario": scenario,
            "method": method,
            "n": len(items),
            "accuracy": float(correct.mean()),
            "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
            "nll": float(np.mean(-np.log(np.clip(chosen, 1e-12, 1.0)))),
            "ece": ece(conf, correct),
            "mean_confidence": float(conf.mean()),
        })
    return sorted(output, key=lambda x: (SCENARIOS.index(x["scenario"]), METHODS.index(x["method"])))


def bootstrap_differences(rows: list[dict], rng: np.random.Generator, reps: int = 2000) -> list[dict]:
    by_key = {(r["scenario"], r["item_id"], r["method"]): r for r in rows}
    output = []
    contrasts = (
        ("dsmt_pcr6", "dst_refined_dempster", "PCR6 versus Dempster"),
        ("dsmt_pcr6_provenance", "dst_refined_dempster_provenance", "PCR6 versus Dempster with matched provenance"),
        ("dsmt_pcr6_provenance", "dsmt_pcr6", "Provenance discount within PCR6"),
    )
    for target, comparator, label in contrasts:
        for scenario in SCENARIOS:
            ids = sorted({r["item_id"] for r in rows if r["scenario"] == scenario})
            diffs = []
            for item_id in ids:
                t = by_key[(scenario, item_id, target)]
                c = by_key[(scenario, item_id, comparator)]
                ti = int(np.argmax([t[f"p_{a}"] for a in ATOMS]) == ATOMS.index(t["truth"]))
                ci = int(np.argmax([c[f"p_{a}"] for a in ATOMS]) == ATOMS.index(c["truth"]))
                diffs.append(ti - ci)
            arr = np.array(diffs, dtype=float)
            samples = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
            output.append({
                "contrast": label,
                "scenario": scenario,
                "target": target,
                "comparator": comparator,
                "accuracy_difference": float(arr.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, summary: list[dict], comparisons: list[dict], config: dict) -> None:
    lookup = {(x["scenario"], x["method"]): x for x in summary}
    lines = [
        "# Controlled DSm-RAG benchmark",
        "",
        "## Scope",
        "",
        "This is a mechanism experiment with synthetic, auditable evidence packets. It tests frame representation, conflict redistribution, and provenance discounting. It does not yet establish performance on real LLM retrieval outputs.",
        "",
        f"Configuration: {config['items_per_scenario']} items per scenario, {config['seeds']} seeds, {len(SCENARIOS)} scenarios, seed base {config['seed_base']}.",
        "",
        "## Accuracy by scenario",
        "",
        "| Scenario | Vote | DST coarse | DST refined | DST refined provenance | Yager refined | DSmT PCR6 | DSmT PCR6 provenance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        vals = [lookup[(scenario, method)]["accuracy"] for method in METHODS]
        lines.append("| " + scenario.replace("_", " ") + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Paired accuracy comparisons", "", "Positive values favor the target named in the contrast.", "", "| Contrast | Scenario | Difference | 95% bootstrap CI |", "|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['contrast']} | {row['scenario'].replace('_', ' ')} | {row['accuracy_difference']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] |")
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- A gain on overlap scenarios supports the nonexclusive or refined representation, but comparison with refined DST shows whether the gain is uniquely attributable to DSmT.",
        "- A gain on partial conflict scenarios supports PCR6 relative to global Dempster normalization.",
        "- A gain confined to duplicated misinformation supports provenance discounting, not PCR6 by itself.",
        "- Performance on independent misinformation tests an important limit: fusion cannot recover truth from a majority of independent, confidently wrong sources without additional reliability information.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python run_experiment.py --items 400 --seeds 8",
        "```",
        "",
        "The complete item-level predictions are stored in `predictions.csv.gz`. Summary metrics and paired bootstrap comparisons are stored as CSV files.",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=400, help="Items per scenario and seed")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=20260914)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    item_id = 0
    for seed_offset in range(args.seeds):
        rng = np.random.default_rng(args.seed_base + seed_offset)
        for scenario in SCENARIOS:
            for _ in range(args.items):
                truth, sources = generate_packet(scenario, rng)
                for method in METHODS:
                    probs = fuse(sources, method)
                    row = {"scenario": scenario, "seed": args.seed_base + seed_offset, "item_id": item_id, "truth": truth, "method": method}
                    row.update({f"p_{atom}": float(probs[i]) for i, atom in enumerate(ATOMS)})
                    rows.append(row)
                item_id += 1

    summary = summarize(rows)
    comparisons = bootstrap_differences(rows, np.random.default_rng(args.seed_base + 999))
    write_csv(args.output / "summary_metrics.csv", summary)
    write_csv(args.output / "paired_bootstrap.csv", comparisons)
    with gzip.open(args.output / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {"items_per_scenario": args.items, "seeds": args.seeds, "seed_base": args.seed_base, "scenarios": SCENARIOS, "methods": METHODS}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(args.output, summary, comparisons, config)
    print(json.dumps({"rows": len(rows), "summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
