# Decision-rule sensitivity analysis

This post-confirmatory analysis responds to Jean Dezert's recommendation to
look beyond the pignistic transformation. It leaves the frozen fusion protocol
unchanged and applies five decision mappings to the same fused BBAs:

- BetP;
- DSmP with epsilon = 0.001;
- maximum singleton belief;
- maximum singleton plausibility;
- minimum Euclidean-family belief-interval distance.

DSmP was implemented for the Shafer model from Dezert and Smarandache's
published formula. A unit test reproduces their two-atom numerical example
to twelve decimal places. The belief-interval-distance criterion implements
the method of Dezert et al. (BELIEF 2016) with the Euclidean-family distance
of Han, Dezert, and Yang (IEEE TSMC: Systems, 2018). For each singleton
decision it compares the fused BBA with the corresponding categorical BBA,
using all nonempty subsets of the frame and their complete [Bel, Pl]
intervals. Minimum BI distance, maximum belief, and maximum plausibility are
ranking rules, not calibrated probability transformations, so Brier scores
are not reported for them. Seventeen implementation tests pass.

## Controlled partial-ignorance condition

| Fusion rule | Decision rule | Accuracy | Brier |
|---|---|---:|---:|
| Dempster | BetP | 0.985 | 0.1878 |
| Dempster | DSmP | 0.981 | 0.1800 |
| Dempster | Minimum BI distance | 0.985 | — |
| Dempster | Max belief | 0.994 | — |
| Dempster | Max plausibility | 0.977 | — |
| PCR6 | BetP | 1.000 | 0.3185 |
| PCR6 | DSmP | 1.000 | 0.2314 |
| PCR6 | Minimum BI distance | 1.000 | — |
| PCR6 | Max belief | 1.000 | — |
| PCR6 | Max plausibility | 1.000 | — |
| PCR6+ | BetP | 1.000 | 0.2733 |
| PCR6+ | DSmP | 1.000 | 0.2575 |
| PCR6+ | Minimum BI distance | 1.000 | — |
| PCR6+ | Max belief | 1.000 | — |
| PCR6+ | Max plausibility | 1.000 | — |

All PCR decisions remained correct. DSmP substantially improved PCR6's Brier
score but only modestly improved PCR6+. Dempster retained the lowest Brier
score in this controlled condition.

Minimum BI distance produced the same top-1 accuracy as BetP for every fusion
rule in the controlled partial-ignorance condition.

## RAMDocs applicability subset

| Fusion rule | Decision rule | Exact set | Set F1 | Brier |
|---|---|---:|---:|---:|
| Dempster | BetP | 0.4885 | 0.7705 | 0.3263 |
| Dempster | DSmP | 0.4846 | 0.7712 | 0.4690 |
| Dempster | Minimum BI distance | 0.4885 | 0.7705 | — |
| Dempster | Max belief | 0.4846 | 0.7705 | — |
| Dempster | Max plausibility | 0.5000 | 0.7756 | — |
| PCR6 | BetP | 0.4923 | 0.7718 | 0.2354 |
| PCR6 | DSmP | 0.4923 | 0.7776 | 0.4389 |
| PCR6 | Minimum BI distance | 0.4923 | 0.7718 | — |
| PCR6 | Max belief | 0.5000 | 0.7833 | — |
| PCR6 | Max plausibility | 0.5000 | 0.7750 | — |
| PCR6+ | BetP | 0.4885 | 0.7718 | 0.2629 |
| PCR6+ | DSmP | 0.4962 | 0.7788 | 0.4279 |
| PCR6+ | Minimum BI distance | 0.4885 | 0.7718 | — |
| PCR6+ | Max belief | 0.5077 | 0.7859 | — |
| PCR6+ | Max plausibility | 0.4962 | 0.7750 | — |

DSmP and maximum belief slightly improved hard answer recovery for PCR6+.
However, DSmP sharply worsened Brier error for every fusion rule, consistent
with a more concentrated transformation applied to NLI-derived masses that
were not calibrated for DSmP. BetP therefore remains the primary probability
mapping for calibration claims, while maximum belief provides the best
exploratory exact-set and set-F1 values in this subset.

Minimum BI distance reproduced the BetP answer sets for Dempster, PCR6, and
PCR6+ on the applicability subset. Thus, it uses the complete belief intervals
but did not change hard answer recovery in this particular NLI-to-BBA setting.

These results reinforce the paper's bounded conclusion: the fusion operator
and the decision transformation must be selected jointly for the downstream
loss. A sharper probability transformation can improve ranking while harming
probabilistic calibration.
