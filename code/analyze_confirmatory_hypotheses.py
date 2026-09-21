"""One-sided paired sign-flip tests and Holm adjustment for protocol v1.4."""

import csv
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
REPS = 100_000
SEED = 20260916


def read(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def paired(rows, selector, metric):
    plus = {int(row["item_id"]): float(row[metric]) for row in rows if row["rule"] == "pcr6_plus" and selector(row)}
    ordinary = {int(row["item_id"]): float(row[metric]) for row in rows if row["rule"] == "pcr6" and selector(row)}
    return np.asarray([plus[index] - ordinary[index] for index in sorted(set(plus) & set(ordinary))])


def sign_flip_pvalue(differences, rng):
    observed = float(differences.mean())
    extreme = 0
    for start in range(0, REPS, 1000):
        size = min(1000, REPS - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        extreme += int(np.sum((signs * differences).mean(axis=1) <= observed + 1e-15))
    return observed, (extreme + 1) / (REPS + 1)


def main():
    controlled = read(RESULTS / "pcr6_plus" / "controlled_item_results.csv")
    ramdocs = read(RESULTS / "ramdocs_pcr6_plus" / "ramdocs_pcr6_plus_item_results.csv")
    tests = [
        ("H1", "Controlled partial-ignorance Brier", paired(controlled, lambda row: row["condition"] == "partial_ignorance_conflict", "brier")),
        ("H2", "Vacuous-source invariance TV", paired(controlled, lambda row: row["condition"] == "vacuous_source_injection", "invariance_tv")),
        ("H3", "RAMDocs applicability Brier", paired(ramdocs, lambda row: row["applicability"] == "True", "brier")),
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for hypothesis, label, differences in tests:
        effect, p = sign_flip_pvalue(differences, rng)
        rows.append({"hypothesis": hypothesis, "label": label, "n": len(differences), "difference_pcr6_plus_minus_pcr6": effect, "one_sided_p": p})

    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["one_sided_p"])
    running = 0.0
    m = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * rows[index]["one_sided_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["reject_at_0_05"] = running <= 0.05

    output = RESULTS / "pcr6_plus" / "confirmatory_hypotheses.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

