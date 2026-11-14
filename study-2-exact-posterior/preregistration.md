# Pre-registration

Written and committed **before** any result was computed. Hypotheses, acceptance criteria
and analysis choices are fixed here so they cannot be adjusted after seeing the numbers.
This file is not edited after the first run; corrections go in `RESULTS.md` as amendments.

## Background

A prior study (see `../inr-repo`) fitted ensembles of implicit neural representations to
synthetic images and tested the aleatoric–epistemic decomposition against causes injected
by construction. It found:

- the aleatoric term recovered the *pattern* of a noise map (rank correlation 0.81) but not
  its *level* (≈ 0.30× true variance in one setting, ≈ 9× in another);
- the epistemic term responded correctly to missing data (33×) and thinned data (16×);
- a purely aleatoric intervention (doubling channel noise) inflated the epistemic term 3.5×;
- the decomposition did **not** beat total predictive variance at predicting failures
  (AUROC 0.81 vs 0.83, bootstrap [−0.056, +0.002]).

Three weaknesses limit what those results can establish:

1. **One data draw per setting.** No error bars, so none of the ratios have a sampling
   distribution attached.
2. **No ground truth for the decomposition itself.** The study could check whether estimates
   responded to injected causes, but not how far they sat from the correct values, because
   the correct values were unknown.
3. **Small ensembles.** At M = 5, the relative standard error of a variance estimate is
   √(2/(M−1)) ≈ 71%. It is unclear how much of the observed failure is simply that.

## What this study changes

**Model class.** Random-feature Bayesian linear regression (RFBLR) instead of a trained
network. This is not an arbitrary substitution: the first layer of the prior study's INR was
*fixed, untrained* random Fourier features, so RFBLR is the same function class with the
output layer solved in closed form rather than by gradient descent.

The point of the substitution is that RFBLR has an **exact posterior**, so the true aleatoric
and epistemic terms are known analytically. The ensemble estimate can then be scored against
ground truth, not only against injected causes.

**Replication.** R = 30 independent data draws per configuration, so every quantity gets a
sampling distribution.

**Ensemble size sweep.** M ∈ {2, 3, 5, 9, 15, 25, 50}, to separate finite-M error from
model error.

**No tuning on the evaluation.** All hyperparameters are fixed in `config.py` before the
first run and are not changed afterwards.

## Hypotheses and acceptance criteria

Stated as predictions with thresholds, decided now.

**H1 — Finite-M explains a large share of epistemic error.**
The relative error of the ensemble epistemic estimate against the exact posterior epistemic
variance will fall approximately as √(2/(M−1)).
*Accept* if the fitted exponent of relative error vs (M−1) is within [−0.65, −0.35]
(theory: −0.5). *Reject* otherwise.

**H2 — Entanglement survives after finite-M error is removed.**
Doubling the true noise level will raise the *exact* epistemic variance by more than 10%,
i.e. the entanglement observed in the prior study is not purely an artefact of small
ensembles.
*Accept* if the median ratio across draws exceeds 1.10 with a 95% bootstrap interval
excluding 1.0.

**H3 — The decomposition does not beat total variance at predicting error, even with exact
terms and adequate power.**
Compare AUROC for identifying the worst-decile-error pixels, using (a) total predictive
variance and (b) the exact epistemic term alone.
*Accept the null* (no advantage) if the 95% bootstrap interval on the AUROC difference
contains 0 with R = 30 draws. Record the interval width either way, since the prior study's
comparison was underpowered.

**H4 — Conformal calibration restores marginal but not conditional coverage.**
With a held-out calibration split, global coverage of a nominal 90% interval will fall within
[0.88, 0.92], while coverage inside an unobserved region will remain below 0.88.
*Accept* if both hold in at least 25 of 30 draws.

## Analysis plan

- All intervals are percentile bootstrap over data draws, 10,000 resamples.
- No result is reported from a single draw.
- Any effect smaller than the across-draw spread is reported as unproven, following the
  standard used in the prior work.
- Failed hypotheses are reported in `RESULTS.md` with the same prominence as confirmed ones.

## Known limitations, stated in advance

- Synthetic 2-D signals only. Nothing here establishes behaviour on real images or 3-D.
- RFBLR is a *linear* model in a fixed feature space. A trained network can adapt its
  features, so the finite-M result transfers but the misspecification result may not.
- The noise model is Gaussian and known in form, which is friendlier than reality.
- This study says nothing about whether any *other* decomposition method works better.
