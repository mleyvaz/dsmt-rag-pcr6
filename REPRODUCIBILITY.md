# Reproducibility Notes

## Experimental isolation

The RAMDocs oracle experiment uses the document-level `answer` field as the output of a perfect claim extractor. It evaluates the fusion layer, not free-form answer extraction. The document labels `correct`, `misinfo` and `noise` are used only to construct declared stress variants and evaluate predictions. They are not passed to the fusion functions.

## Matched comparison

Provenance-aware PCR6 and provenance-aware Dempster combination receive identical candidate answers, entity assignments, confidence scores and duplicate discounting. The comparison therefore isolates the combination rule within the implemented answer frame.

## Provenance control

Exact normalized document text defines a provenance group. If `g` copies share confidence `r`, each receives effective confidence

`r_eff = 1 - (1-r)^(1/g)`

so that their combined residual ignorance equals that of one original source.

## Deterministic analysis

- Oracle RAMDocs paired bootstrap seed: `20260914`
- NLI F1 paired bootstrap seed: `20260915`
- NLI Brier paired bootstrap seed: `20260916`
- Oracle bootstrap replicates: `2000`
- NLI bootstrap replicates: `4000`

## Important limits

- The RAMDocs answer frame is exclusive and does not test the full hyper-power-set representation of DSmT.
- NLI candidates are enumerated from the RAMDocs document answer field.
- The duplicate stress condition uses exact copies, not paraphrases or hidden common sources.
- Secondary comparisons are exploratory and are not adjusted for multiplicity.

