# Secondary analyses

Written after seeing the primary results, and labelled as exploratory throughout. These establish whether the three failed pre-registered hypotheses reflect real properties of the decomposition or artefacts of this study's design.

## S1 - Why the epistemic error does not shrink with M

*(This section's control was rebuilt once. The first attempt compared the main design against a bootstrap-resample ensemble, which turned out to be a second biased construction rather than a reference point, so it could not isolate anything. The reference used below is an ensemble of exact posterior samples, for which the estimator is unbiased by construction.)*

H1 predicted the relative error of the ensemble epistemic estimate would fall as `(M-1)^-0.5`. The fitted exponent was -0.102. Either the estimator is converging to something other than the posterior epistemic variance, or there is a bug.

If the members are draws from one posterior, the spread of their means estimates that posterior's variance and the error is pure finite-sample noise, vanishing as M grows. If the members instead differ systematically -- here, each carries its own random Fourier feature draw, exactly as each network in a deep ensemble carries its own initialisation -- then the spread of their means estimates the variance across *model families*, which does not converge to the posterior epistemic variance however large M gets.

Fitting `rel_err(M) = a*(M-1)^b + c` separates the two: `a` is the part that decays, `c` is the floor that does not.

Fitted floor **c = 0.543**, decaying part `a = 0.311`, exponent on the decaying part **b = -0.605** (residual SSE 1.51e-04).

So the error is not shrinking because roughly 64% of it at M = 2, and 95% of it at M = 50, is a floor that no amount of ensembling removes.

**Direct test of the mechanism.** Three ensembles are built on the same data and scored the same way, differing only in what makes their members differ:

1. **posterior samples** -- weights drawn from the exact posterior. The sample variance of their predictions is an unbiased estimator of the epistemic variance, so this is the reference: any error here is pure finite-M noise and must decay as `(M-1)^-0.5`.
2. **feature draws** -- the main design, and the analogue of a deep ensemble whose members differ by initialisation.
3. **data resamples** -- one shared feature draw, members differing by a bootstrap resample of the observations.

| M | posterior samples | feature draws (main) | data resamples |
|---|---|---|---|
| 2 | 0.820 | 0.853 | 0.995 |
| 5 | 0.468 | 0.673 | 3.615 |
| 9 | 0.334 | 0.635 | 4.957 |
| 25 | 0.199 | 0.589 | 5.575 |
| 50 | 0.140 | 0.574 | 5.724 |

Fitted exponent, posterior samples: **-0.457** -- against the theoretical -0.5, and against -0.102 for the main design. (10 draws, 40s.)

Asymptotic bias, measured as the median ratio of estimate to truth at M = 50: **0.44x** for feature draws, **6.72x** for data resamples, **1.00x** for posterior samples.

**The mechanism is confirmed, and it is not a bug.** Given members that really are posterior samples, the estimator behaves exactly as theory says: the error decays at the predicted rate and converges on the truth. The floor in the main design comes from the members, not the estimator. An ensemble whose members differ by their random feature draw -- the standard construction, and the one an INR ensemble uses -- does not sample the posterior. Its spread measures disagreement between model families, which is a different quantity, and adding members only buys precision on that different quantity. This is why H1 failed: it assumed the gap between estimate and truth was sampling error that more members would close, and most of it is not.

The two biased constructions miss in opposite directions -- feature draws understate the epistemic variance, data resamples overstate it by several times -- so there is no correction factor that a practitioner could apply without already knowing the answer.


## S2 - H2's accepted ratio, against what the model predicts analytically

H2 was accepted: doubling the noise multiplied the exact epistemic variance by 3.63. The pre-registered threshold was 1.10, which in hindsight was far too lax to be informative. The honest question is whether 3.63 is a finding or arithmetic.

For this model the posterior covariance is `(Phi' W Phi + alpha I)^-1` with `W = diag(1/sigma^2)`. Doubling every noise standard deviation divides `W` by 4, so in the data-dominated regime the covariance -- and with it every epistemic variance -- is multiplied by very nearly 4, falling slightly short because the prior precision `alpha` does not scale.

Measured: **3.629** [3.613, 3.645]. Predicted ceiling: 4.000.

So the measurement matches theory, and the correct reading of H2 is the stronger and less flattering one: the epistemic term's dependence on the noise level is not an estimation artefact *and not a surprise either*. It is what the decomposition is defined to do. An intervention that changes only the aleatoric noise changes the epistemic term by a factor of four. Any claim that these two quantities separate a model's ignorance from the world's randomness is wrong at the level of the definition, not at the level of the estimator -- which is what the baseline study could not distinguish.


## S3 - H3 depends on which error you ask the decomposition to predict

The null in H3 was that the exact epistemic term is no better than total predictive variance at finding high-error pixels. It was rejected: epistemic alone won by +0.026 AUROC. But the error being predicted was the deviation of the fit from the *clean signal*, and the aleatoric term describes observation noise, which is by construction irrelevant to that target. Adding it can only dilute the signal.

The practitioner's question is usually the other one: how far will the next *observation* fall from the prediction. Below, both targets are scored with the same predictors, using the ensemble prediction at the baseline study's M = 9.

| target | total variance | epistemic only | aleatoric only | epistemic - total |
|---|---|---|---|---|
| clean signal | 0.798 | 0.826 | 0.545 | +0.028 [+0.023, +0.034] |
| noisy observation | 0.801 | 0.822 | 0.568 | +0.021 [+0.016, +0.027] |

**The sign does not flip.** Epistemic variance alone beats total variance on both targets (+0.028 against the clean signal, +0.021 against the noisy observation), and the margin narrows by only 0.007 when the aleatoric term becomes relevant to the question. The prediction behind H3 -- that the split would earn nothing -- is wrong in a modest but consistent direction.

What does change with the target is how much the aleatoric term knows on its own: 0.545 AUROC against the clean signal, where it should be uninformative, rising to 0.568 against the noisy observation, where it is the right quantity. That it is above chance on the first target at all is a property of this scene rather than of the decomposition: the noise level rises left to right and the fit is also worse on the right, for reasons of sampling density, so the two are correlated in the ground truth. A reader should take the small margins here as specific to this geometry, not as a general ranking.

The operational reading is narrower than H3 anticipated but still useful: if the question is where a fit has drifted, the epistemic term is the better detector and adding the aleatoric term dilutes it slightly; if the question is where the next measurement will land, total variance is the better-calibrated quantity even though it ranks pixels marginally worse. Ranking and calibration are different jobs, and a single AUROC answers only the first.


## S4 - Why H4 failed, and what it was actually measuring

H4 predicted conformal scaling would restore global coverage to 0.88-0.92 while leaving the unobserved block below 0.88, in at least 25 of 30 draws. Global coverage landed at 0.900, exactly as predicted, in nearly every draw. The joint condition failed on the hole term: mean coverage inside the hole was 0.822, but it varied so much across draws that it sat above 0.88 in several of them.

- global condition alone held in **27/30** draws
- hole condition alone held in **13/30** draws
- hole coverage across draws: min 0.242, median 0.902, max 0.969, sd 0.178

The shape of that distribution matters more than its mean, and it is the reason H4 was the wrong prediction to make. Median hole coverage is 0.902 -- essentially nominal -- while the mean is 0.822 and the minimum is 0.242. Conditional coverage is not reliably *low*; it is reliably *unpredictable*. In roughly half the draws the unobserved block happened to be covered about as well as everywhere else, and in the rest it was short by up to two thirds.

That is a worse property than uniform under-coverage, and a more honest headline than the one H4 was written to produce. Uniform under-coverage could be corrected with a margin. A quantity whose conditional coverage varies from 0.24 to 0.97 across statistically identical draws cannot be corrected by any fixed factor, and the marginal number -- a tidy 0.900, exactly as promised, in 27 of 30 draws -- carries no information about which case you are in. Conformal calibration does what it guarantees. The error is in reading a marginal guarantee as if it said something about a region.

We had pre-registered the direction of the failure rather than its variance, which is a design error on our side and not a surprise about conformal prediction.

### What drives the swing

The scene, the hole and its contents are identical in every draw. Only the sampling mask and the noise change. So the swing has to come from one of three places: the conformal quantile, the predicted uncertainty inside the hole, or the actual error inside the hole. Each is measured below against hole coverage.

| quantity | spread across draws | correlation with hole coverage |
|---|---|---|
| conformal quantile q | 2.577 to 3.539 (1.4x) | -0.040 [-0.321, +0.223] |
| predicted sd inside hole | 0.083 to 0.157 (1.9x) | -0.013 [-0.307, +0.255] |
| actual error inside hole | 0.061 to 0.453 (7.4x) | -0.746 [-0.902, -0.408] |

The coverage swing is driven almost entirely by the actual error (r = -0.75 [-0.90, -0.42]), and the predicted uncertainty tracks it not at all (r = -0.01). The reason is visible in the spreads: across draws the true extrapolation error inside the hole varies by a factor of 7.4, while the uncertainty the model predicts there varies by only 1.9 and the conformal quantile by 1.4.

**This is the result worth reporting.** In the region the model never observed, the uncertainty estimate is close to constant while the error it is supposed to describe varies sevenfold. It is not merely miscalibrated by some factor -- it has almost no mutual information with how wrong the extrapolation actually is, and the exact Bayesian posterior does not fix this, because the posterior's variance in an unobserved region is set by the prior and the geometry of the sampling, not by what the unseen signal happens to do. Conformal calibration rescales a quantity that was uninformative there to begin with, which is why it buys a correct marginal number and no regional guarantee.
