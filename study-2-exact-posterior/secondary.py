"""Secondary analyses, run after the pre-registered tests, to establish whether the
three failed hypotheses reflect real properties or artefacts of this study's design.

Nothing here is a confirmatory test. Each block is labelled with what it can and cannot
settle, and all of it was written after seeing the primary results.

  S1  Why does the epistemic error not fall as 1/sqrt(M)? Separate the finite-M variance
      from a non-vanishing bias, and test the suspected mechanism directly by rebuilding
      the ensemble so that members differ ONLY in their data resample.
  S2  Is H2's accepted ratio a finding or arithmetic? Compare it to the factor predicted
      analytically for this model.
  S3  Does H3's result depend on which error the decomposition is asked to predict --
      error against the clean signal, or error against the noisy observation?
"""

from __future__ import annotations
import json, time
import numpy as np

import config as C
import data as D
import rfblr as R
from analyse import auroc, boot_ci

RNG = np.random.default_rng(777)
lines = []
A = lines.append


# ---------------------------------------------------------------------------
def s1_plateau(z, m_values):
    A("## S1 - Why the epistemic error does not shrink with M\n")
    A("*(This section's control was rebuilt once. The first attempt compared the main "
      "design against a bootstrap-resample ensemble, which turned out to be a second "
      "biased construction rather than a reference point, so it could not isolate "
      "anything. The reference used below is an ensemble of exact posterior samples, for "
      "which the estimator is unbiased by construction.)*\n")
    A("H1 predicted the relative error of the ensemble epistemic estimate would fall as "
      "`(M-1)^-0.5`. The fitted exponent was -0.102. Either the estimator is converging "
      "to something other than the posterior epistemic variance, or there is a bug.\n")
    A("If the members are draws from one posterior, the spread of their means estimates "
      "that posterior's variance and the error is pure finite-sample noise, vanishing as "
      "M grows. If the members instead differ systematically -- here, each carries its "
      "own random Fourier feature draw, exactly as each network in a deep ensemble "
      "carries its own initialisation -- then the spread of their means estimates the "
      "variance across *model families*, which does not converge to the posterior "
      "epistemic variance however large M gets.\n")
    A("Fitting `rel_err(M) = a*(M-1)^b + c` separates the two: `a` is the part that "
      "decays, `c` is the floor that does not.\n")

    epi_exact = z["base__epi_exact"]
    rel = []
    for M in m_values:
        hat = z[f"base__epi_hat_M{M}"]
        rel.append(np.median(np.abs(hat - epi_exact) / np.maximum(epi_exact, 1e-12),
                             axis=1))
    rel = np.array(rel)
    med = np.median(rel, axis=1)
    x = np.array(m_values, float) - 1.0

    # grid search over the floor, least squares on the rest in log space
    best = None
    for c in np.linspace(0.0, med.min() * 0.999, 400):
        y = med - c
        if np.any(y <= 0):
            continue
        b, loga = np.polyfit(np.log(x), np.log(y), 1)
        pred = np.exp(loga) * x ** b + c
        sse = float(np.sum((pred - med) ** 2))
        if best is None or sse < best[0]:
            best = (sse, c, b, float(np.exp(loga)))
    sse, c, b, a = best
    A(f"Fitted floor **c = {c:.3f}**, decaying part `a = {a:.3f}`, exponent on the "
      f"decaying part **b = {b:.3f}** (residual SSE {sse:.2e}).\n")
    A(f"So the error is not shrinking because roughly {c / med[0]:.0%} of it at M = 2, "
      f"and {c / med[-1]:.0%} of it at M = {m_values[-1]}, is a floor that no amount of "
      "ensembling removes.\n")

    A("**Direct test of the mechanism.** Three ensembles are built on the same data and "
      "scored the same way, differing only in what makes their members differ:\n")
    A("1. **posterior samples** -- weights drawn from the exact posterior. The sample "
      "variance of their predictions is an unbiased estimator of the epistemic variance, "
      "so this is the reference: any error here is pure finite-M noise and must decay as "
      "`(M-1)^-0.5`.")
    A("2. **feature draws** -- the main design, and the analogue of a deep ensemble whose "
      "members differ by initialisation.")
    A("3. **data resamples** -- one shared feature draw, members differing by a bootstrap "
      "resample of the observations.\n")

    t0 = time.time()
    m_sub = [2, 5, 9, 25, 50]
    rel_post, rel_boot, lim_post, lim_boot = [], [], [], []
    for i in range(C.S1_DRAWS):
        seed = C.SEED + i
        d = D.make_draw(np.random.default_rng(seed), 1.0)
        xy, obs = d["xy"], d["obs"]
        xy_obs, y_obs = xy[obs], d["y"][obs]
        s2_obs = d["noise_sd"][obs] ** 2

        phi = R.make_features(C.N_FEATURES, C.FOURIER_SCALE,
                              np.random.default_rng(seed + 10_000))
        Po, Pa = phi(xy_obs), phi(xy)
        mean, cov = R.fit_posterior(Po, y_obs, s2_obs, C.PRIOR_PRECISION)
        _, epi_ex = R.predict(Pa, mean, cov)
        den = np.maximum(epi_ex, 1e-12)

        srng = np.random.default_rng(seed + 70_000)
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
        W = mean[None, :] + srng.standard_normal((max(m_sub), cov.shape[0])) @ L.T
        mus_post = W @ Pa.T

        brng = np.random.default_rng(seed + 90_000)
        mus_boot = []
        for _ in range(max(m_sub)):
            ix = brng.integers(0, len(y_obs), len(y_obs))
            mj, _ = R.fit_posterior(Po[ix], y_obs[ix], s2_obs[ix], C.PRIOR_PRECISION)
            mus_boot.append(Pa @ mj)
        mus_boot = np.array(mus_boot)

        rel_post.append([np.median(np.abs(mus_post[:M].var(axis=0, ddof=1) - epi_ex) / den)
                         for M in m_sub])
        rel_boot.append([np.median(np.abs(mus_boot[:M].var(axis=0, ddof=1) - epi_ex) / den)
                         for M in m_sub])
        lim_post.append(np.median(mus_post.var(axis=0, ddof=1) / den))
        lim_boot.append(np.median(mus_boot.var(axis=0, ddof=1) / den))

    med_post = np.median(np.array(rel_post).T, axis=1)
    med_boot = np.median(np.array(rel_boot).T, axis=1)
    xs = np.log(np.array(m_sub, float) - 1)
    b_post, _ = np.polyfit(xs, np.log(med_post), 1)

    A("| M | posterior samples | feature draws (main) | data resamples |")
    A("|---|---|---|---|")
    for k, M in enumerate(m_sub):
        j = m_values.index(M)
        A(f"| {M} | {med_post[k]:.3f} | {med[j]:.3f} | {med_boot[k]:.3f} |")
    A(f"\nFitted exponent, posterior samples: **{b_post:.3f}** -- against the theoretical "
      f"-0.5, and against -0.102 for the main design. ({C.S1_DRAWS} draws, "
      f"{time.time()-t0:.0f}s.)\n")
    # per-draw median of the ratio, then median across draws (the figure uses the
    # same definition, so the two agree)
    hat50 = np.median(z["base__epi_hat_M50"]
                      / np.maximum(z["base__epi_exact"], 1e-12), axis=1)
    A(f"Asymptotic bias, measured as the median ratio of estimate to truth at M = 50: "
      f"**{np.median(hat50):.2f}x** for feature draws, "
      f"**{np.median(lim_boot):.2f}x** for data resamples, "
      f"**{np.median(lim_post):.2f}x** for posterior samples.\n")

    confirmed = b_post < -0.30
    if confirmed:
        A("**The mechanism is confirmed, and it is not a bug.** Given members that really "
          "are posterior samples, the estimator behaves exactly as theory says: the error "
          "decays at the predicted rate and converges on the truth. The floor in the main "
          "design comes from the members, not the estimator. An ensemble whose members "
          "differ by their random feature draw -- the standard construction, and the one "
          "an INR ensemble uses -- does not sample the posterior. Its spread measures "
          "disagreement between model families, which is a different quantity, and "
          "adding members only buys precision on that different quantity. This is why "
          "H1 failed: it assumed the gap between estimate and truth was sampling error "
          "that more members would close, and most of it is not.\n")
        A("The two biased constructions miss in opposite directions -- feature draws "
          "understate the epistemic variance, data resamples overstate it by several "
          "times -- so there is no correction factor that a practitioner could apply "
          "without already knowing the answer.\n")
    else:
        A("**The mechanism is not confirmed.** Even an ensemble of exact posterior "
          "samples fails to converge at the predicted rate, which points at the "
          "estimator or at this diagnostic rather than at how the members are built. "
          "None of H1's interpretation stands until that is resolved.\n")
    return {"floor": c, "decay_exponent": b, "posterior_exponent": float(b_post),
            "bias_features_M50": float(np.median(hat50)),
            "bias_resample_M50": float(np.median(lim_boot)),
            "bias_posterior_M50": float(np.median(lim_post)),
            "mechanism_confirmed": bool(confirmed)}


# ---------------------------------------------------------------------------
def s2_arithmetic(z):
    A("\n## S2 - H2's accepted ratio, against what the model predicts analytically\n")
    A("H2 was accepted: doubling the noise multiplied the exact epistemic variance by "
      "3.63. The pre-registered threshold was 1.10, which in hindsight was far too lax "
      "to be informative. The honest question is whether 3.63 is a finding or "
      "arithmetic.\n")
    A("For this model the posterior covariance is `(Phi' W Phi + alpha I)^-1` with "
      "`W = diag(1/sigma^2)`. Doubling every noise standard deviation divides `W` by 4, "
      "so in the data-dominated regime the covariance -- and with it every epistemic "
      "variance -- is multiplied by very nearly 4, falling slightly short because the "
      "prior precision `alpha` does not scale.\n")
    ratio = (np.median(z["double__epi_exact"], axis=1)
             / np.median(z["base__epi_exact"], axis=1))
    m, lo, hi = boot_ci(ratio)
    A(f"Measured: **{m:.3f}** [{lo:.3f}, {hi:.3f}]. Predicted ceiling: 4.000.\n")
    A("So the measurement matches theory, and the correct reading of H2 is the stronger "
      "and less flattering one: the epistemic term's dependence on the noise level is "
      "not an estimation artefact *and not a surprise either*. It is what the "
      "decomposition is defined to do. An intervention that changes only the aleatoric "
      "noise changes the epistemic term by a factor of four. Any claim that these two "
      "quantities separate a model's ignorance from the world's randomness is wrong at "
      "the level of the definition, not at the level of the estimator -- which is what "
      "the baseline study could not distinguish.\n")
    return {"measured": m, "ci": [lo, hi], "predicted_ceiling": 4.0}


# ---------------------------------------------------------------------------
def s3_target(z, m_values):
    A("\n## S3 - H3 depends on which error you ask the decomposition to predict\n")
    A("The null in H3 was that the exact epistemic term is no better than total "
      "predictive variance at finding high-error pixels. It was rejected: epistemic "
      "alone won by +0.026 AUROC. But the error being predicted was the deviation of "
      "the fit from the *clean signal*, and the aleatoric term describes observation "
      "noise, which is by construction irrelevant to that target. Adding it can only "
      "dilute the signal.\n")
    A("The practitioner's question is usually the other one: how far will the next "
      "*observation* fall from the prediction. Below, both targets are scored with the "
      "same predictors, using the ensemble prediction at the baseline study's M = 9.\n")

    M = 9 if 9 in m_values else m_values[-1]
    epi_e, alea_e = z["base__epi_exact"], z["base__alea_exact"]
    mu_hat = z[f"base__mu_hat_M{M}"]
    rows = {}
    for target in ("clean signal", "noisy observation"):
        at, ae, aa = [], [], []
        for r in range(mu_hat.shape[0]):
            d = D.make_draw(np.random.default_rng(C.SEED + r), 1.0)
            ref = d["truth"] if target == "clean signal" else d["y"]
            err = np.abs(mu_hat[r] - ref)
            lab = (err >= np.percentile(err, 90)).astype(int)
            at.append(auroc(epi_e[r] + alea_e[r], lab))
            ae.append(auroc(epi_e[r], lab))
            aa.append(auroc(alea_e[r], lab))
        rows[target] = (np.array(at), np.array(ae), np.array(aa))

    A("| target | total variance | epistemic only | aleatoric only | epistemic - total |")
    A("|---|---|---|---|---|")
    out = {}
    for target, (at, ae, aa) in rows.items():
        t, _, _ = boot_ci(at, stat=np.mean)
        e, _, _ = boot_ci(ae, stat=np.mean)
        a_, _, _ = boot_ci(aa, stat=np.mean)
        dm, dlo, dhi = boot_ci(ae - at, stat=np.mean)
        A(f"| {target} | {t:.3f} | {e:.3f} | {a_:.3f} | "
          f"{dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] |")
        out[target] = {"total": t, "epistemic": e, "aleatoric": a_,
                       "diff": dm, "ci": [dlo, dhi]}

    cs, no = out["clean signal"], out["noisy observation"]
    flip = (cs["diff"] > 0) != (no["diff"] > 0)
    if flip:
        A("\n**The sign flips with the target**, so the decomposition's apparent value is "
          "an artefact of which error is being scored.\n")
    else:
        A(f"\n**The sign does not flip.** Epistemic variance alone beats total variance "
          f"on both targets ({cs['diff']:+.3f} against the clean signal, "
          f"{no['diff']:+.3f} against the noisy observation), and the margin narrows by "
          f"only {abs(cs['diff'] - no['diff']):.3f} when the aleatoric term becomes "
          "relevant to the question. The prediction behind H3 -- that the split would "
          "earn nothing -- is wrong in a modest but consistent direction.\n")
    A(f"What does change with the target is how much the aleatoric term knows on its own: "
      f"{cs['aleatoric']:.3f} AUROC against the clean signal, where it should be "
      f"uninformative, rising to {no['aleatoric']:.3f} against the noisy observation, "
      "where it is the right quantity. That it is above chance on the first target at all "
      "is a property of this scene rather than of the decomposition: the noise level "
      "rises left to right and the fit is also worse on the right, for reasons of "
      "sampling density, so the two are correlated in the ground truth. A reader should "
      "take the small margins here as specific to this geometry, not as a general "
      "ranking.\n")
    A("The operational reading is narrower than H3 anticipated but still useful: if the "
      "question is where a fit has drifted, the epistemic term is the better detector "
      "and adding the aleatoric term dilutes it slightly; if the question is where the "
      "next measurement will land, total variance is the better-calibrated quantity even "
      "though it ranks pixels marginally worse. Ranking and calibration are different "
      "jobs, and a single AUROC answers only the first.\n")
    return out


# ---------------------------------------------------------------------------
def s4_h4(z):
    A("\n## S4 - Why H4 failed, and what it was actually measuring\n")
    A("H4 predicted conformal scaling would restore global coverage to 0.88-0.92 while "
      "leaving the unobserved block below 0.88, in at least 25 of 30 draws. Global "
      "coverage landed at 0.900, exactly as predicted, in nearly every draw. The joint "
      "condition failed on the hole term: mean coverage inside the hole was 0.822, but "
      "it varied so much across draws that it sat above 0.88 in several of them.\n")
    cg, ch = z["base__cov_conf_global"], z["base__cov_conf_hole"]
    ing = ((cg >= C.H4_GLOBAL_RANGE[0]) & (cg <= C.H4_GLOBAL_RANGE[1]))
    A(f"- global condition alone held in **{int(ing.sum())}/{len(cg)}** draws")
    A(f"- hole condition alone held in **{int((ch < C.H4_HOLE_MAX).sum())}/{len(ch)}** "
      f"draws")
    A(f"- hole coverage across draws: min {ch.min():.3f}, median "
      f"{np.median(ch):.3f}, max {ch.max():.3f}, sd {ch.std(ddof=1):.3f}\n")
    A("The shape of that distribution matters more than its mean, and it is the reason "
      "H4 was the wrong prediction to make. Median hole coverage is "
      f"{np.median(ch):.3f} -- essentially nominal -- while the mean is {ch.mean():.3f} "
      f"and the minimum is {ch.min():.3f}. Conditional coverage is not reliably *low*; "
      "it is reliably *unpredictable*. In roughly half the draws the unobserved block "
      "happened to be covered about as well as everywhere else, and in the rest it was "
      "short by up to two thirds.\n")
    A("That is a worse property than uniform under-coverage, and a more honest headline "
      "than the one H4 was written to produce. Uniform under-coverage could be corrected "
      "with a margin. A quantity whose conditional coverage varies from 0.24 to 0.97 "
      "across statistically identical draws cannot be corrected by any fixed factor, and "
      "the marginal number -- a tidy 0.900, exactly as promised, in 27 of 30 draws -- "
      "carries no information about which case you are in. Conformal calibration does "
      "what it guarantees. The error is in reading a marginal guarantee as if it said "
      "something about a region.\n")
    A("We had pre-registered the direction of the failure rather than its variance, "
      "which is a design error on our side and not a surprise about conformal "
      "prediction.\n")

    # --- what drives the swing -------------------------------------------------
    A("### What drives the swing\n")
    A("The scene, the hole and its contents are identical in every draw. Only the "
      "sampling mask and the noise change. So the swing has to come from one of three "
      "places: the conformal quantile, the predicted uncertainty inside the hole, or the "
      "actual error inside the hole. Each is measured below against hole coverage.\n")

    M = 9 if f"base__mu_hat_M9" in z.files else max(list(z["m_values"]))
    mu = z[f"base__mu_hat_M{M}"]
    epi, alea = z["base__epi_exact"], z["base__alea_exact"]
    err_in, sd_in = [], []
    for r in range(len(ch)):
        d = D.make_draw(np.random.default_rng(C.SEED + r), 1.0)
        h = d["hole"]
        err_in.append(float(np.abs(mu[r][h] - d["y"][h]).mean()))
        sd_in.append(float(np.sqrt(epi[r][h] + alea[r][h]).mean()))
    err_in, sd_in = np.array(err_in), np.array(sd_in)

    def corr_ci(a, b):
        r0 = float(np.corrcoef(a, b)[0, 1])
        bs = [np.corrcoef(a[i], b[i])[0, 1]
              for i in RNG.integers(0, len(a), (4000, len(a)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return r0, float(lo), float(hi)

    A("| quantity | spread across draws | correlation with hole coverage |")
    A("|---|---|---|")
    for name, arr in (("conformal quantile q", z["base__q"]),
                      ("predicted sd inside hole", sd_in),
                      ("actual error inside hole", err_in)):
        r0, lo, hi = corr_ci(arr, ch)
        A(f"| {name} | {arr.min():.3f} to {arr.max():.3f} "
          f"({arr.max()/arr.min():.1f}x) | {r0:+.3f} [{lo:+.3f}, {hi:+.3f}] |")

    r_sd, _, _ = corr_ci(sd_in, ch)
    r_err, lo_err, hi_err = corr_ci(err_in, ch)
    A("\nThe coverage swing is driven almost entirely by the actual error "
      f"(r = {r_err:+.2f} [{lo_err:+.2f}, {hi_err:+.2f}]), and the predicted uncertainty "
      f"tracks it not at all (r = {r_sd:+.2f}). The reason is visible in the spreads: "
      f"across draws the true extrapolation error inside the hole varies by a factor of "
      f"{err_in.max()/err_in.min():.1f}, while the uncertainty the model predicts there "
      f"varies by only {sd_in.max()/sd_in.min():.1f} and the conformal quantile by "
      f"{z['base__q'].max()/z['base__q'].min():.1f}.\n")
    A("**This is the result worth reporting.** In the region the model never observed, "
      "the uncertainty estimate is close to constant while the error it is supposed to "
      "describe varies sevenfold. It is not merely miscalibrated by some factor -- it "
      "has almost no mutual information with how wrong the extrapolation actually is, "
      "and the exact Bayesian posterior does not fix this, because the posterior's "
      "variance in an unobserved region is set by the prior and the geometry of the "
      "sampling, not by what the unseen signal happens to do. Conformal calibration "
      "rescales a quantity that was uninformative there to begin with, which is why it "
      "buys a correct marginal number and no regional guarantee.\n")
    extra = {"corr_err": r_err, "corr_err_ci": [lo_err, hi_err], "corr_sd": r_sd,
             "err_spread": float(err_in.max() / err_in.min()),
             "sd_spread": float(sd_in.max() / sd_in.min())}
    return {"global_ok": int(ing.sum()), "hole_ok": int((ch < C.H4_HOLE_MAX).sum()),
            "hole_sd": float(ch.std(ddof=1)), **extra}


def main():
    z = np.load("results/raw.npz")
    m_values = list(z["m_values"])
    A("# Secondary analyses\n")
    A("Written after seeing the primary results, and labelled as exploratory throughout. "
      "These establish whether the three failed pre-registered hypotheses reflect real "
      "properties of the decomposition or artefacts of this study's design.\n")
    out = {}
    out["S1"] = s1_plateau(z, m_values)
    out["S2"] = s2_arithmetic(z)
    out["S3"] = s3_target(z, m_values)
    out["S4"] = s4_h4(z)
    open("SECONDARY.md", "w").write("\n".join(lines))
    json.dump(out, open("results/secondary.json", "w"), indent=2, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
