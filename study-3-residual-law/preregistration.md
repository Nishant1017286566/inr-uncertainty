# Pre-registration — causal test of the residual law

Fixed **before** these experiments were run. The observational finding being tested comes
from the 97-setting grid of the v3 study (and replicates in v1 and v2):

> Across 97 settings spanning 679x in the aleatoric estimate `A`, the data collapse onto
> `log(A/sigma^2) = 1.864 * log(train_rmse/sigma) - 0.396`, R^2 = 0.932.
> By contrast `corr(log sigma^2, log A) = +0.354`, and `A` lands within 1.5x of the true
> noise variance in 13% of the grid.

That is a correlation across settings that differ in data volume, capacity and noise at the
same time. It is consistent with "the aleatoric head reports the training residual, not the
noise", but it does not establish it. Two interventions can.

## The mechanism under test

`M1` — The Gaussian NLL head minimises `0.5*(log v + r^2/v)`, whose optimum is `v = r^2`,
the squared residual **on the training data**. It therefore reports the training residual
whatever produced it, and equals the noise variance only when the model is fitted to exactly
the noise floor — neither memorising below it nor underfitting above it.

If `M1` is right, then *any* intervention that moves the training residual while leaving the
true noise untouched must move `A` by the square of that movement, and the three sweeps of
the original study are one phenomenon rather than three.

## Interventions

Both hold the noise level, the image family, the ensemble size and the evaluation fixed.

**I1 — task difficulty (`DIVERSITY` = 0, 0.5, 1, 2, 4).** The `DIVERSITY` knob scales how
much the images differ from one another. Raising it makes the shared network's job harder at
*identical* capacity, data volume and noise. This knob exists in the original code and was
never swept.

**I2 — optimisation budget (`STEPS` = 150, 300, 600, 1000, 2000, 4000).** Changes nothing
about the data, the model, or the noise — only how far the fit is taken. The purest
intervention available: if `A` tracks the training residual, `A` must fall as training
proceeds, in a world where the true aleatoric uncertainty is constant by construction.

Baseline for both: `n=10, width=50, noise=10` — the original study's own baseline, so these
runs drop into its grid. 3 seeds per setting, full fidelity (`G=32, M=5`).

## Hypotheses and decision rules

| | Prediction | Accept if |
|---|---|---|
| **H1** | `A` rises with `DIVERSITY` while `sigma^2` is fixed | Spearman rho(DIVERSITY, A) > 0.7 and A's range exceeds 2x |
| **H2** | `A` falls as `STEPS` rises | Spearman rho(STEPS, A) < -0.7 |
| **H3** | Both interventions obey the same law as the grid: regressing `log(A/sigma^2)` on `log(train_rmse/sigma)` over the pooled intervention runs gives the slope the grid gave | fitted slope in [1.4, 2.4] and R^2 > 0.80 |
| **H4** | The law is **not** explained by the error at test points instead: `train_rmse` beats `err` as a predictor of `A` on the intervention runs | `train_rmse^2` has the lower median multiplicative error of the two |
| **H5** *(falsifier)* | The epistemic term `E` does **not** obey a corresponding law — it is not a residual meter, so no slope-2 relation holds for it | abs(slope) < 1.0 for `log(E/sigma^2)` on `log(train_rmse/sigma)`, or R^2 < 0.5 |

H5 is the one that can embarrass the story: if `E` tracks the training residual just as well,
then the finding is "both heads track the residual" and the aleatoric-specific mechanism `M1`
is not what is driving it.

## What would falsify `M1`

- `A` unmoved by `I2` (H2 fails): the head is not reading the residual, and the grid
  correlation is confounded by something that co-varies with capacity and data volume.
- The pooled slope far from 2 (H3 fails): the relation is not the NLL optimum.
- H5 fails: nothing aleatoric-specific is happening.

Any failure is reported in the same detail as a success. Seeds, settings and all raw outputs
are saved to `results/`.
