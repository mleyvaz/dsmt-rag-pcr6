# Provenance Aware PCR6 for Conflicting Evidence in RAG

This repository contains preliminary code and derived results for a matched comparison of provenance-aware PCR6 and Dempster-Shafer combination on conflicting retrieval-augmented generation evidence.

## Main preliminary result

On 500 RAMDocs questions with three additional exact copies of every misinformation document, provenance-aware PCR6 reached 80.6% exact-set accuracy and 92.31% set F1. Provenance-matched Dempster combination reached 78.2% exact-set accuracy and 90.98% set F1. The paired F1 difference was 1.33 percentage points with a 95% bootstrap interval from 0.60 to 2.33 points.

With identical NLI-derived masses, PCR6 and Dempster selected the same answer sets. PCR6 nevertheless reduced Brier error by 0.0353 in the original condition and by 0.0190 under misinformation dominance. Both paired bootstrap intervals excluded zero.

These are conditional findings. They do not show that DSmT or PCR6 is universally superior to Dempster-Shafer theory.

## Repository contents

- `code/` contains the fusion implementations, dataset experiments, NLI scoring script, configuration and unit tests.
- `results/ramdocs/` contains aggregate, bootstrap and item-level oracle-fusion results.
- `results/conflictqa/` contains aggregate, bootstrap and item-level negative-control results.
- `results/nli/` contains NLI scores, metadata, fusion results and paired bootstrap analyses.
- `docs/DATA_SOURCES.md` records dataset provenance, retrieval dates, licenses and SHA-256 hashes.
- `docs/RESULTADOS_RAG_CONSOLIDADOS_ES.md` provides a Spanish interpretation of the results.

## Data policy

Raw RAMDocs and ConflictQA files are not redistributed here. Download them from their official sources and verify the hashes in `docs/DATA_SOURCES.md`. The repository contains derived outputs needed to audit the reported metrics.

## Environment used

- Python 3.12.14
- NumPy 2.3.5
- Node.js 24.19.0
- `@huggingface/transformers` 4.2.0
- ONNX Runtime through Transformers.js
- NLI model `Xenova/distilbert-base-uncased-mnli`
- NLI dtype `q8`
- Maximum NLI sequence length 256 tokens

## Quick verification

From the repository root:

```powershell
python -m unittest discover -s code -p "test_*.py"
python code/verify_repository.py
```

To rerun RAMDocs after downloading the official test file:

```powershell
python code/run_ramdocs_experiment.py --data PATH_TO_RAMDocs_test.jsonl --output results/ramdocs_reproduced
```

To regenerate NLI scores:

```powershell
npm install
node code/run_nli_scoring.mjs --data PATH_TO_RAMDocs_test.jsonl --output results/nli_reproduced/nli_scores.jsonl
python code/run_ramdocs_nli_fusion.py --data PATH_TO_RAMDocs_test.jsonl --nli results/nli_reproduced/nli_scores.jsonl --output results/nli_reproduced
```

The ConflictQA runner exposes its current command-line interface with:

```powershell
python code/run_conflictqa_experiment.py --help
```

## Reproducibility status

This is a preliminary research release. The code, derived outputs and deterministic seeds are included. Before journal submission, the repository should receive a permanent archive identifier and an explicit software license selected by the authors.

## Authorship status

Authorship order and contributions remain subject to confirmation by all participating researchers. Repository availability does not itself establish authorship.
