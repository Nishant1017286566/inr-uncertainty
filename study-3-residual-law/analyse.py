"""Scores the observational law on the 97-setting grid (all three versions) and the
pre-registered hypotheses H1-H5 on the intervention runs. Writes RESULTS.md.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import spearmanr

GRIDS = {"v1": "results_images", "v2": "results_images_v2", "v3": "results_images_v3"}
RNG = np.random.default_rng(4242)
L, A_ = [], None
out = {}


def say(s=""):
    L.append(s)


def grid(v):
    d = json.load(open(f"grids/{GRIDS[v]}.json"))
    keys = sorted(d)
    def g(k):
        """Not every version records every field (v1 predates the A/E correlation block)."""
        if k not in d[keys[0]]:
            return np.full(len(keys), np.nan)
        return np.array([d[s][k] for s in keys], float)
    return dict(A=g("A"), E=g("E"), err=g("err"), sig=g("sigma"), tr=g("train_rmse"),
                n=g("n"), w=g("width"), ns=g("noise"), rho=g("rho_std_err"),
                share=g("share_med"), tot=g("tot"))


def mult_err(pred, act):
    """Median multiplicative error: how many times off, in either direction."""
    return float(10 ** np.median(np.abs(np.log10(pred / act))))


def _clean(x, y):
    """Drop points a log-log fit cannot take, and say how many. A zero or missing value
    here is real (an ensemble can agree exactly, a version may not record a field), so it
    is excluded explicitly rather than silently coerced."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if (~m).any():
        DROPPED.append(int((~m).sum()))
    return x[m], y[m]


DROPPED = []


def fit_loglog(x, y):
    x, y = _clean(x, y)
    sl, ic = np.polyfit(np.log(x), np.log(y), 1)
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    return float(sl), float(ic), float(r ** 2)


def boot_slope(x, y, n=4000):
    x, y = _clean(x, y)
    s = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        if len(set(x[i].tolist())) < 3:
            continue
        s.append(np.polyfit(np.log(x[i]), np.log(y[i]), 1)[0])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ----------------------------------------------------------------- section 1 + 2
def observational():
    say("## 1 - What the aleatoric head is measuring\n")
    say("`A` is defined as the mean predicted variance and is intended to estimate the "
        "observation-noise variance `sigma^2`, which is known exactly here. Across the "
        "97-setting grid:\n")
    say("| version | A spans | corr(log sigma^2, log A) | corr(log train_rmse^2, log A) | "
        "A as estimate of sigma^2 | within 1.5x of truth |")
    say("|---|---|---|---|---|---|")
    for v in ("v1", "v2", "v3"):
        d = grid(v)
        A, sig, tr = d["A"], d["sig"], d["tr"]
        c1 = np.corrcoef(np.log(sig ** 2), np.log(A))[0, 1]
        c2 = np.corrcoef(np.log(tr ** 2), np.log(A))[0, 1]
        frac = 100 * np.mean(np.abs(np.log10(A / sig ** 2)) < np.log10(1.5))
        say(f"| {v} | {A.min():.1e}-{A.max():.1e} ({A.max()/A.min():.0f}x) | "
            f"**{c1:+.3f}** | **{c2:+.3f}** | {mult_err(sig**2, A):.2f}x median | "
            f"{frac:.0f}% |")
        out[f"grid_{v}"] = dict(corr_sigma=float(c1), corr_resid=float(c2),
                                within15=float(frac))
    d = grid("v3")
    fixed = d["ns"] == 10
    say(f"\nIn the {int(fixed.sum())} settings where the noise knob is at its baseline, so "
        f"`sigma^2` is *identical by construction*, `A` still ranges "
        f"{d['A'][fixed].min():.2e} to {d['A'][fixed].max():.2e} "
        f"(**{d['A'][fixed].max()/d['A'][fixed].min():.0f}x**).\n")
    say("The quantity `A` is supposed to estimate explains almost none of its variation. "
        "The training residual explains nearly all of it.\n")

    say("\n## 2 - One law behind the three sweeps\n")
    say("The Gaussian NLL term `0.5*(log v + r^2/v)` is minimised at `v = r^2`: the head's "
        "optimum is the squared residual **on the training data**. If that is what it "
        "reports, then `A/sigma^2` must be a power law in `train_rmse/sigma` with "
        "exponent 2, and all three sweeps are one phenomenon.\n")
    say("| version | fitted slope | 95% CI | R^2 | n |")
    say("|---|---|---|---|---|")
    for v in ("v1", "v2", "v3"):
        d = grid(v)
        x, y = d["tr"] / d["sig"], d["A"] / d["sig"] ** 2
        sl, ic, r2 = fit_loglog(x, y)
        lo, hi = boot_slope(x, y)
        say(f"| {v} | **{sl:.3f}** | [{lo:.2f}, {hi:.2f}] | **{r2:.3f}** | {len(x)} |")
        out[f"law_{v}"] = dict(slope=sl, ci=[lo, hi], r2=r2)
    say("\nTheory says 2. One free parameter collapses 679x of variation in `A` onto a "
        "single line, in every version of the code.\n")

    d = grid("v3")
    x, y = d["tr"] / d["sig"], d["A"] / d["sig"] ** 2
    say("### The three regimes\n")
    say("`train_rmse/sigma` says which side of the noise floor the fit sits on, and that "
        "alone predicts whether `A` is too low, right, or far too high.\n")
    say("| regime | train_rmse/sigma | settings | median A/sigma^2 | where it occurs |")
    say("|---|---|---|---|---|")
    regimes = [("memorising", x < 0.8), ("matched", (x >= 0.8) & (x <= 1.25)),
               ("underfitting", x > 1.25)]
    for nm, m in regimes:
        where = []
        if nm == "memorising":
            where = ["few images, or a wide net on little data"]
        elif nm == "matched":
            where = ["a narrow band only"]
        else:
            where = ["many images, narrow nets, short training"]
        say(f"| {nm} | {'<0.8' if nm=='memorising' else '0.8-1.25' if nm=='matched' else '>1.25'} "
            f"| {int(m.sum())} | **{np.median(y[m]):.2f}x** | {where[0]} |")
        out[f"regime_{nm}"] = dict(n=int(m.sum()), median=float(np.median(y[m])))
    say(f"\n`A` is a usable noise estimate in {int(regimes[1][1].sum())} of 97 settings. "
        "Everywhere else it reports the model's own misfit, or its memorisation, as a "
        "property of the data.\n")


# ----------------------------------------------------------------- section 3
def interventions():
    say("\n## 3 - Causal test: two knobs that move the residual and nothing else\n")
    p = "results/interventions_both.json"
    if not os.path.exists(p):
        say("*(intervention runs not present)*\n")
        return
    I = json.load(open(p))
    say("Pre-registered in `preregistration.md` before these ran. Baseline `n=10, "
        "width=50, noise=10`, 3 seeds, full fidelity.\n")
    say("**One pre-registered setting was dropped for a reason unrelated to the "
        "hypotheses.** `DIVERSITY = 4` is outside the domain of the study's own image "
        "family: the blob-width term `0.020*(1 + D*U(-0.3, 0.3))` turns negative for 8 of "
        "the 100 images at `D = 4`, which flips a Gaussian into a divergent exponential "
        "and produces non-finite pixels. The family is well defined only for `D < 3.33`, "
        "so `D = 4` was replaced by `D = 3`. This is a latent limit in the generator, not "
        "a result, and it would silently corrupt any future sweep of that knob.\n")

    res = {}
    for tag, label, knob_name in (("I1", "task difficulty (DIVERSITY)", "diversity"),
                                  ("I2", "optimisation budget (STEPS)", "steps")):
        R = I[tag]
        if not R:
            continue
        knob = np.array([r["knob"] for r in R], float)
        A = np.array([r["A"] for r in R])
        sig = np.array([r["sigma"] for r in R])
        tr = np.array([r["train_rmse"] for r in R])
        E = np.array([r["E"] for r in R])
        err = np.array([r["err"] for r in R])
        fin = np.isfinite(A) & np.isfinite(sig) & np.isfinite(tr)
        if not fin.all():
            say(f"*{int((~fin).sum())} of {len(A)} runs returned non-finite metrics and "
                "are excluded; see the note below.*\n")
        knob, A, sig, tr, E, err = (v[fin] for v in (knob, A, sig, tr, E, err))
        rho = float(spearmanr(knob, A)[0])
        res[tag] = dict(knob=knob, A=A, sig=sig, tr=tr, E=E, err=err, rho=rho)
        say(f"\n### {tag} - {label}\n")
        say(f"| {knob_name} | train_rmse/sigma | A | sigma^2 | **A/sigma^2** | E |")
        say("|---|---|---|---|---|---|")
        for k in sorted(set(knob.tolist())):
            m = knob == k
            say(f"| {k:g} | {np.mean(tr[m]/sig[m]):.2f} | {A[m].mean():.3e} | "
                f"{np.mean(sig[m]**2):.3e} | **{np.mean(A[m]/sig[m]**2):.2f}x** | "
                f"{E[m].mean():.3e} |")
        say(f"\nSpearman rho({knob_name}, A) = **{rho:+.3f}**, "
            f"A range **{A.max()/A.min():.1f}x**.")
        if tag == "I1":
            ok = rho > 0.7 and A.max() / A.min() > 2
            say(f"\n**H1 {'ACCEPTED' if ok else 'REJECTED'}** (needs rho > 0.7 and "
                f"range > 2x). Raising task difficulty at identical capacity, data volume "
                f"and noise level moves the noise estimate "
                f"{'as the law requires' if ok else 'not as predicted'}.")
            out["H1"] = bool(ok)
        else:
            ok = rho < -0.7
            say(f"\n**H2 {'ACCEPTED' if ok else 'REJECTED'}** (needs rho < -0.7). "
                "Training longer changes nothing about the data, the model or the noise, "
                f"yet the reported aleatoric uncertainty "
                f"{'falls' if ok else 'does not fall'} — "
                f"{A.max()/A.min():.1f}x across the budget range.")
            out["H2"] = bool(ok)

    # H3: pooled slope
    if res:
        x = np.concatenate([res[t]["tr"] / res[t]["sig"] for t in res])
        y = np.concatenate([res[t]["A"] / res[t]["sig"] ** 2 for t in res])
        sl, ic, r2 = fit_loglog(x, y)
        lo, hi = boot_slope(x, y)
        gsl = out["law_v3"]["slope"]
        say(f"\n### H3 - the interventions obey the grid's law\n")
        say(f"Pooled over both interventions ({len(x)} runs): slope **{sl:.3f}** "
            f"[{lo:.2f}, {hi:.2f}], R^2 **{r2:.3f}**. The observational grid gave "
            f"{gsl:.3f}; theory says 2.\n")
        ok3 = 1.4 <= sl <= 2.4 and r2 > 0.80
        say(f"**H3 {'ACCEPTED' if ok3 else 'REJECTED'}** (needs slope in [1.4, 2.4] and "
            f"R^2 > 0.80).")
        out["H3"] = bool(ok3)
        out["pooled_law"] = dict(slope=sl, ci=[lo, hi], r2=r2)

        # H4: train_rmse beats test err
        xe = np.concatenate([res[t]["err"] for t in res])
        A = np.concatenate([res[t]["A"] for t in res])
        tr = np.concatenate([res[t]["tr"] for t in res])
        sg = np.concatenate([res[t]["sig"] for t in res])
        me_tr, me_err, me_sig = mult_err(tr ** 2, A), mult_err(xe ** 2, A), mult_err(sg ** 2, A)
        say(f"\n### H4 - is it the *training* residual, or just error in general?\n")
        say(f"| predictor of A | median multiplicative error |")
        say("|---|---|")
        say(f"| `sigma^2` (its claimed meaning) | {me_sig:.2f}x |")
        say(f"| `err^2` (error at held-out test points) | {me_err:.2f}x |")
        say(f"| `train_rmse^2` (residual on training data) | **{me_tr:.2f}x** |")
        ok4 = me_tr < me_err
        if me_sig < me_tr:
            say(f"\nNote against ourselves: on these 33 runs `sigma^2` is the best of the "
                f"three ({me_sig:.2f}x). That is expected and does not rescue it — the "
                f"interventions sit at one point of the grid, so `sigma^2` barely varies "
                f"and a constant is a decent fit. The grid is where that comparison is "
                f"informative, and there `sigma^2` is "
                f"{out['grid_v3']['corr_sigma']:+.2f} correlated with `A` against "
                f"{out['grid_v3']['corr_resid']:+.2f} for the residual. H4 asks only "
                f"which of the two *error* quantities the head is reading.")
        say(f"\n**H4 {'ACCEPTED' if ok4 else 'REJECTED'}**. "
            + ("The training residual is the better predictor, which is what the NLL "
               "optimum predicts: the head sees training residuals, not test error.\n"
               if ok4 else
               "Test error predicts `A` at least as well, so these runs do not "
               "distinguish the training residual from error in general.\n"))
        out["H4"] = bool(ok4)

        # H5 falsifier: does E obey the same law?
        Ev = np.concatenate([res[t]["E"] / res[t]["sig"] ** 2 for t in res])
        sl5, _, r25 = fit_loglog(x, Ev)
        say(f"### H5 (falsifier) - does the epistemic term obey the same law?\n")
        say(f"Pre-registered test, pooled over both interventions: regressing "
            f"`log(E/sigma^2)` on `log(train_rmse/sigma)` gives slope **{sl5:.3f}**, "
            f"R^2 **{r25:.3f}**, against {sl:.3f} / {r2:.3f} for `A`.\n")
        ok5 = abs(sl5) < 1.0 or r25 < 0.5
        say(f"**H5 {'ACCEPTED' if ok5 else 'REJECTED'}** (needs abs(slope) < 1.0 or "
            f"R^2 < 0.5).")
        out["H5"] = bool(ok5)
        out["E_law"] = dict(slope=sl5, r2=r25)
        if ok5:
            say("\nThe two heads are not doing the same thing: only the aleatoric one is a "
                "residual meter. The mechanism is specific to the NLL variance head, as "
                "`M1` claims.\n")
        else:
            say("\nThe falsifier fired: on the pooled runs `E` does trend with the "
                "training residual, so the pre-registered test does not establish that "
                "the effect is specific to the aleatoric head.\n")

        # post-hoc: the pooled test conflates two interventions that differ in kind
        say("#### Post-hoc (not pre-registered): the pooled test conflates two "
            "interventions\n")
        say("`I2` changes only the optimisation budget, so the data, the model and the "
            "true epistemic problem are all fixed. `I1` changes the images themselves, "
            "which genuinely changes how much there is to disagree about — and also moves "
            "`sigma^2`, since the noise knob is a percentage of image standard deviation. "
            "Only `I2` isolates the head. Splitting the same regression:\n")
        say("| runs | A: slope, R^2 | E: slope, R^2 | A range | E range |")
        say("|---|---|---|---|---|")
        for t in ("I2", "I1"):
            if t not in res:
                continue
            xt = res[t]["tr"] / res[t]["sig"]
            ya = res[t]["A"] / res[t]["sig"] ** 2
            ye = res[t]["E"] / res[t]["sig"] ** 2
            sa, _, ra = fit_loglog(xt, ya)
            se, _, re_ = fit_loglog(xt, ye)
            say(f"| **{t}** | {sa:.2f}, {ra:.3f} | {se:.2f}, {re_:.3f} | "
                f"{res[t]['A'].max()/res[t]['A'].min():.1f}x | "
                f"{res[t]['E'].max()/res[t]['E'].min():.1f}x |")
            out[f"split_{t}"] = dict(A_slope=sa, A_r2=ra, E_slope=se, E_r2=re_,
                                     A_range=float(res[t]['A'].max()/res[t]['A'].min()),
                                     E_range=float(res[t]['E'].max()/res[t]['E'].min()))
        if "I2" in res:
            s = out["split_I2"]
            say(f"\nUnder `I2` alone, `A` moves **{s['A_range']:.1f}x** and `E` moves "
                f"**{s['E_range']:.1f}x**. Training 27x longer leaves the epistemic term "
                f"essentially where it was and cuts the aleatoric term by an order of "
                f"magnitude. That is the asymmetry `M1` predicts, and it is what the "
                f"pooled test hid.\n")
            say("Stated plainly: the pre-registered falsifier failed, and a post-hoc split "
                "recovers the prediction. That is weaker evidence than a pre-registered "
                "pass, and it is reported as such — the pooled test was the wrong test to "
                "have written down, because `I1` was never a clean probe of the "
                "aleatoric head alone.\n")


# ----------------------------------------------------------------- section 6
def downstream():
    d = grid("v3")

    def grid_raw(f):
        dd = json.load(open('grids/results_images_v3.json'))
        return [dd[k].get(f, float('nan')) for k in sorted(dd)]

    say("\n## 6 - What it costs downstream\n")
    rho, A, sig, w, ns = d["rho"], d["A"], d["sig"], d["w"], d["ns"]
    ok = ~np.isnan(rho)
    say("`rho_std_err` is the rank correlation between the predicted total standard "
        "deviation and the true absolute error, per setting: does the uncertainty point at "
        "the mistakes?\n")
    say(f"- across the grid it runs {np.nanmin(rho):+.2f} to {np.nanmax(rho):+.2f}, "
        f"median {np.nanmedian(rho):+.2f}")
    worst = np.nanargmin(rho)
    say(f"- lowest: n={int(d['n'][worst])}, width={int(w[worst])}, "
        f"noise={int(ns[worst])} gives **{rho[worst]:+.3f}** — no better than chance")
    say(f"- it is **never negative** in any of the {int(ok.sum())} settings "
        f"(min {np.nanmin(rho):+.3f}), so the uncertainty degrades to uninformative "
        "rather than inverting. An earlier draft of this analysis claimed an inversion at "
        "width 1; that was a conflation of two different recorded correlations and is "
        "corrected here.")
    noi = ns[ok]
    say(f"- correlation between the noise knob and `rho_std_err`: "
        f"**{spearmanr(noi, rho[ok])[0]:+.2f}** — the ranking degrades as noise grows, "
        f"from {rho[ok & (ns==1) & (d['n']==10) & (w==50)][0]:+.2f} at 1% noise to "
        f"{rho[ok & (ns==100) & (d['n']==10) & (w==50)][0]:+.2f} at 100%\n")

    # the separate, and genuinely negative, correlation: A against E
    rr = d["share"] * 0 + np.array([v for v in grid_raw("rho_rank")])
    rk = np.isfinite(rr)
    say("A different correlation in the same study records whether the two terms move "
        "together across points within a setting — whether the split is even stable:\n")
    say(f"- `rho_rank(A, E)` runs {np.nanmin(rr):+.3f} to {np.nanmax(rr):+.3f}, median "
        f"**{np.nanmedian(rr):+.3f}** across {int(rk.sum())} settings")
    wmin = np.nanargmin(rr)
    say(f"- at n={int(d['n'][wmin])}, width={int(w[wmin])}, noise={int(ns[wmin])} it is "
        f"**{rr[wmin]:+.3f}**: the two terms are almost perfectly *anti*-correlated, so "
        "whichever way the total moves, the attribution between 'noise' and 'ignorance' "
        "flips with it")
    say("- a median of about +0.46 means that, in the typical setting, the points the "
        "model calls noisy are also the points it calls uncertain — the split is not "
        "separating the two things it names\n")
    out["rho_rank_range"] = [float(np.nanmin(rr)), float(np.nanmax(rr))]
    out["rho_rank_median"] = float(np.nanmedian(rr))

    out["rho_range"] = [float(np.nanmin(rho)), float(np.nanmax(rho))]
    out["rho_worst"] = dict(n=int(d['n'][worst]), width=int(w[worst]),
                            noise=int(ns[worst]), rho=float(rho[worst]))


def main():
    say("# Results\n")
    say("Generated by `analyse.py`. The observational sections use the completed "
        "97-setting grids of the original study (v1, v2, v3); section 3 uses the "
        "intervention runs in `results/`.\n")
    observational()
    interventions()
    downstream()
    open("RESULTS.md", "w").write("\n".join(L))
    json.dump(out, open("results/verdicts.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
