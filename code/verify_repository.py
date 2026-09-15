from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        ROOT / "results" / "ramdocs" / "ramdocs_summary.csv",
        ROOT / "results" / "ramdocs" / "ramdocs_bootstrap.csv",
        ROOT / "results" / "conflictqa" / "conflictqa_summary.csv",
        ROOT / "results" / "nli" / "nli_summary.csv",
        ROOT / "results" / "nli" / "nli_brier_bootstrap.csv",
        ROOT / "results" / "nli" / "nli_scores.jsonl.meta.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    ram = read_csv(required[0])
    target = next(row for row in ram if row["variant"] == "duplicate_misinfo_x3" and row["method"] == "dsmt_pcr6_provenance")
    control = next(row for row in ram if row["variant"] == "duplicate_misinfo_x3" and row["method"] == "dst_factorized_provenance")
    assert abs(float(target["exact_set"]) - 0.806) < 1e-12
    assert abs(float(target["set_f1"]) - 0.9231333333333333) < 1e-12
    assert abs(float(control["exact_set"]) - 0.782) < 1e-12
    assert abs(float(control["set_f1"]) - 0.9098) < 1e-12

    boot = read_csv(required[1])
    comparison = next(row for row in boot if row["variant"] == "duplicate_misinfo_x3" and row["target"] == "dsmt_pcr6_provenance" and row["comparator"] == "dst_factorized_provenance")
    assert float(comparison["ci95_low"]) > 0
    assert float(comparison["ci95_high"]) > 0

    brier = read_csv(required[4])
    for row in brier:
        if row["target"] == "dsmt_nli_pcr6_provenance" and row["comparator"] == "dst_nli_factorized_provenance":
            assert float(row["ci95_low"]) < 0
            assert float(row["ci95_high"]) < 0

    with required[5].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["pairs"] == 7802
    assert metadata["dtype"] == "q8"

    print("Repository verification passed")
    print("Primary exact-set result: 0.806 versus 0.782")
    print("Primary set-F1 result: 0.923133 versus 0.909800")
    print("NLI pairs: 7802")


if __name__ == "__main__":
    main()

