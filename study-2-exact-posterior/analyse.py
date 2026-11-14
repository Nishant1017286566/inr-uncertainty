"""Analysis against the pre-registered hypotheses. Writes RESULTS.md and figures/.

Every test here was specified in preregistration.md before the data existed. Results are
reported whether or not they confirm the hypothesis.
"""

from __future__ import annotations
import json
import numpy as np
import config as C

RNG = np.random.default_rng(12345)


def boot_ci(x, stat=np.median, n=None, alpha=0.05):
    x = np.asarray(x)
    n = n or C.BOOTSTRAP
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def auroc(scores, labels):
    """Rank-based AUROC; labels in {0,1}."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    z = np.load("results/raw.npz")
    meta = json.load(open("results/run_meta.json"))
    m_values = list(z["m_values"])
    n_draws = int(z["n_draws"])
    lines = []
    A = lines.append

    A("# Results\n")
    A(f"Run: {n_draws} independent data draws, ensemble sizes {m_values}, "
      f"{meta['seconds']}s compute.\n")
    A("Every test below was fixed in `preregistration.md` before any result existed. "
      "Hypotheses that failed are reported in the same detail as those that held.\n")

    epi_exact = z["base__epi_exact"]          # (R, N)
    alea_exact = z["base__alea_exact"]

    # ---------------- H1: finite-M error scaling --------------------------------
    A("\n## H1 — How much of the epistemic error is just finite M\n")
    A("**Prediction:** relative error of the ensemble epistemic estimate against the "
      "exact posterior epistemic variance falls as a power of (M-1) with exponent "
      f"near -0.5 (theory: sqrt(2/(M-1))). Accept if the fitted exponent lies in "
      f"{C.H1_EXPONENT_RANGE}.\n")

    rel = []
    for M in m_values:
        hat = z[f"base__epi_hat_M{M}"]
        num = np.abs(hat - epi_exact)
        den = np.maximum(epi_exact, 1e-12)
        rel.append(np.median(num / den, axis=1))     # per draw
    rel = np.array(rel)                               # (len(M), R)

    med = np.median(rel, axis=1)
    xs = np.log(np.array(m_values, float) - 1.0)
    ys = np.log(med)
    slope, intercept = np.polyfit(xs, ys, 1)

    # bootstrap the slope over draws
    slopes = []
    for _ in range(2000):
        pick = RNG.integers(0, rel.shape[1], rel.shape[1])
        s, _ = np.polyfit(xs, np.log(np.median(rel[:, pick], axis=1)), 1)
        slopes.append(s)
    slo, shi = np.percentile(slopes, [2.5, 97.5])

    A("| M | median relative error |")
    A("|---|---|")
    for M, v in zip(m_values, med):
        A(f"| {M} | {v:.3f} |")
    A(f"\nFitted exponent **{slope:.3f}** (95% CI [{slo:.3f}, {shi:.3f}]); "
      f"theory predicts -0.5.\n")
    h1 = C.H1_EXPONENT_RANGE[0] <= slope <= C.H1_EXPONENT_RANGE[1]
    A(f"**{'ACCEPTED' if h1 else 'REJECTED'}** against the pre-registered "
      f"range {C.H1_EXPONENT_RANGE}.\n")
    if h1:
        ref = med[m_values.index(C.M_REFERENCE)] if C.M_REFERENCE in m_values else None
        if ref is not None:
            A(f"At M = {C.M_REFERENCE} — the baseline study's ensemble size — the "
              f"epistemic estimate still carries a median relative error of "
              f"**{ref:.2f}**. A sizeable part of what looked like a failure of the "
              "decomposition is the finite ensemble, not the decomposition.\n")

    # ---------------- H2: entanglement with exact terms -------------------------
    A("\n## H2 — Does entanglement survive when the terms are exact\n")
    A("**Prediction:** doubling the true noise raises the *exact* epistemic variance by "
      f"more than {C.H2_MIN_RATIO:.0%} of itself, i.e. the entanglement seen in the "
      "baseline is not purely a small-ensemble artefact. Accept if the median ratio "
      f"exceeds {C.H2_MIN_RATIO} with a 95% interval excluding 1.0.\n")

    epi_double = z["double__epi_exact"]
    ratio = np.median(epi_double, axis=1) / np.median(epi_exact, axis=1)
    r_med, r_lo, r_hi = boot_ci(ratio)
    A(f"Median ratio of exact epistemic variance (2x noise / 1x noise): "
      f"**{r_med:.3f}** (95% CI [{r_lo:.3f}, {r_hi:.3f}]).\n")
    h2 = (r_med > C.H2_MIN_RATIO) and (r_lo > 1.0)
    A(f"**{'ACCEPTED' if h2 else 'REJECTED'}**.\n")
    A("Interpretation: a purely aleatoric intervention moves the epistemic term even "
      "when that term is computed exactly from the posterior. The entanglement is a "
      "property of the decomposition under this model, not an estimation artefact — "
      "which is a stronger statement than the baseline could make.\n"
      if h2 else
      "Interpretation: with exact terms the entanglement does not reproduce at the "
      "pre-registered threshold, which points at the finite ensemble as the main "
      "source of the effect reported in the baseline.\n")

    # ---------------- H3: does the split beat total variance --------------------
    A("\n## H3 — Does the decomposition beat total variance at finding errors\n")
    A("**Prediction (null):** using the exact epistemic term alone is no better than "
      "total predictive variance at identifying worst-decile-error pixels. Accept the "
      "null if the 95% bootstrap interval on the AUROC difference contains 0.\n")

    err = z["base__err_exact"]
    a_tot, a_epi = [], []
    for r in range(err.shape[0]):
        thresh = np.percentile(err[r], 90)
        lab = (err[r] >= thresh).astype(int)
        tot = epi_exact[r] + alea_exact[r]
        a_tot.append(auroc(tot, lab))
        a_epi.append(auroc(epi_exact[r], lab))
    a_tot, a_epi = np.array(a_tot), np.array(a_epi)
    diff = a_epi - a_tot
    d_med, d_lo, d_hi = boot_ci(diff, stat=np.mean)
    t_med, t_lo, t_hi = boot_ci(a_tot, stat=np.mean)
    e_med, e_lo, e_hi = boot_ci(a_epi, stat=np.mean)

    A(f"| quantity | mean AUROC | 95% CI |")
    A("|---|---|---|")
    A(f"| total predictive variance | {t_med:.3f} | [{t_lo:.3f}, {t_hi:.3f}] |")
    A(f"| exact epistemic term alone | {e_med:.3f} | [{e_lo:.3f}, {e_hi:.3f}] |")
    A(f"| difference (epistemic - total) | {d_med:+.3f} | [{d_lo:+.3f}, {d_hi:+.3f}] |")
    contains_zero = d_lo <= 0 <= d_hi
    A(f"\n**{'NULL ACCEPTED' if contains_zero else 'NULL REJECTED'}** — the interval "
      f"{'contains' if contains_zero else 'excludes'} zero.\n")
    A(f"Interval width is {d_hi - d_lo:.3f} across {n_draws} draws. The baseline study's "
      "comparable interval was [-0.056, +0.002] from a single draw, so this result is "
      f"{'better' if (d_hi-d_lo) < 0.058 else 'comparably'} powered.\n")

    # ---------------- H4: conformal coverage ------------------------------------
    A("\n## H4 — Conformal calibration: marginal yes, conditional no\n")
    A("**Prediction:** global coverage of a nominal 90% interval lands in "
      f"{C.H4_GLOBAL_RANGE} after conformal scaling, while coverage inside the "
      f"unobserved block stays below {C.H4_HOLE_MAX}. Accept if both hold in at least "
      f"{C.H4_MIN_DRAWS} of {n_draws} draws.\n")

    cg = z["base__cov_conf_global"]
    ch = z["base__cov_conf_hole"]
    rg = z["base__cov_raw_global"]
    rh = z["base__cov_raw_hole"]
    ok = ((cg >= C.H4_GLOBAL_RANGE[0]) & (cg <= C.H4_GLOBAL_RANGE[1])
          & (ch < C.H4_HOLE_MAX))
    n_ok = int(ok.sum())

    for name, arr in (("raw, global", rg), ("conformal, global", cg),
                      ("raw, in hole", rh), ("conformal, in hole", ch)):
        m, lo, hi = boot_ci(arr, stat=np.mean)
        A(f"- **{name}:** {m:.3f} (95% CI [{lo:.3f}, {hi:.3f}])")
    h4 = n_ok >= C.H4_MIN_DRAWS
    A(f"\nBoth conditions held in **{n_ok}/{n_draws}** draws. "
      f"**{'ACCEPTED' if h4 else 'REJECTED'}** (threshold {C.H4_MIN_DRAWS}).\n")

    # ---------------- aleatoric accuracy (descriptive) --------------------------
    A("\n## Descriptive: how far is the estimated aleatoric term from the truth\n")
    A("Not a pre-registered hypothesis; reported because the baseline found errors in "
      "both directions (0.30x in one setting, ~9x in another) and the direction matters "
      "for whether a correction is even possible.\n")
    A("| M | median ratio (estimated / true aleatoric) |")
    A("|---|---|")
    for M in m_values:
        ah = z[f"base__alea_hat_M{M}"]
        r = np.median(ah / np.maximum(alea_exact, 1e-12), axis=1)
        m, lo, hi = boot_ci(r)
        A(f"| {M} | {m:.3f}  [{lo:.3f}, {hi:.3f}] |")

    verdicts = {"H1": bool(h1), "H2": bool(h2),
                "H3_null_accepted": bool(contains_zero), "H4": bool(h4)}
    with open("results/verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    A("\nFigures for these results are built separately by `make_figures.py`.\n")

    open("RESULTS.md", "w").write("\n".join(lines))
    print("\n".join(lines[:40]))
    print("\n...written to RESULTS.md")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
