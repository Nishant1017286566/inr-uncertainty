# Results

Run: 30 independent data draws, ensemble sizes [np.int64(2), np.int64(3), np.int64(5), np.int64(9), np.int64(15), np.int64(25), np.int64(50)], 886.5s compute.

Every test below was fixed in `preregistration.md` before any result existed. Hypotheses that failed are reported in the same detail as those that held.


## H1 — How much of the epistemic error is just finite M

**Prediction:** relative error of the ensemble epistemic estimate against the exact posterior epistemic variance falls as a power of (M-1) with exponent near -0.5 (theory: sqrt(2/(M-1))). Accept if the fitted exponent lies in (-0.65, -0.35).

| M | median relative error |
|---|---|
| 2 | 0.853 |
| 3 | 0.756 |
| 5 | 0.673 |
| 9 | 0.635 |
| 15 | 0.600 |
| 25 | 0.589 |
| 50 | 0.574 |

Fitted exponent **-0.102** (95% CI [-0.133, -0.090]); theory predicts -0.5.

**REJECTED** against the pre-registered range (-0.65, -0.35).


## H2 — Does entanglement survive when the terms are exact

**Prediction:** doubling the true noise raises the *exact* epistemic variance by more than 110% of itself, i.e. the entanglement seen in the baseline is not purely a small-ensemble artefact. Accept if the median ratio exceeds 1.1 with a 95% interval excluding 1.0.

Median ratio of exact epistemic variance (2x noise / 1x noise): **3.629** (95% CI [3.613, 3.646]).

**ACCEPTED**.

Interpretation: a purely aleatoric intervention moves the epistemic term even when that term is computed exactly from the posterior. The entanglement is a property of the decomposition under this model, not an estimation artefact — which is a stronger statement than the baseline could make.


## H3 — Does the decomposition beat total variance at finding errors

**Prediction (null):** using the exact epistemic term alone is no better than total predictive variance at identifying worst-decile-error pixels. Accept the null if the 95% bootstrap interval on the AUROC difference contains 0.

| quantity | mean AUROC | 95% CI |
|---|---|---|
| total predictive variance | 0.841 | [0.805, 0.876] |
| exact epistemic term alone | 0.867 | [0.834, 0.898] |
| difference (epistemic - total) | +0.026 | [+0.020, +0.033] |

**NULL REJECTED** — the interval excludes zero.

Interval width is 0.013 across 30 draws. The baseline study's comparable interval was [-0.056, +0.002] from a single draw, so this result is better powered.


## H4 — Conformal calibration: marginal yes, conditional no

**Prediction:** global coverage of a nominal 90% interval lands in (0.88, 0.92) after conformal scaling, while coverage inside the unobserved block stays below 0.88. Accept if both hold in at least 25 of 30 draws.

- **raw, global:** 0.707 (95% CI [0.695, 0.719])
- **conformal, global:** 0.900 (95% CI [0.896, 0.905])
- **raw, in hole:** 0.554 (95% CI [0.469, 0.638])
- **conformal, in hole:** 0.822 (95% CI [0.754, 0.879])

Both conditions held in **13/30** draws. **REJECTED** (threshold 25).


## Descriptive: how far is the estimated aleatoric term from the truth

Not a pre-registered hypothesis; reported because the baseline found errors in both directions (0.30x in one setting, ~9x in another) and the direction matters for whether a correction is even possible.

| M | median ratio (estimated / true aleatoric) |
|---|---|
| 2 | 1.312  [1.227, 1.401] |
| 3 | 1.336  [1.279, 1.421] |
| 5 | 1.378  [1.312, 1.485] |
| 9 | 1.406  [1.346, 1.532] |
| 15 | 1.415  [1.349, 1.546] |
| 25 | 1.422  [1.345, 1.523] |
| 50 | 1.432  [1.349, 1.540] |

Figures for these results are built separately by `make_figures.py`.
