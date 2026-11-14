"""Figures for the write-up. Recomputes the small S1 ensembles (cached) and reads
results/raw.npz for everything else.

Palette: three categorical hues validated for colour-vision deficiency separation
(worst adjacent CVD dE 9.2, normal-vision dE 27.6). Every series is also directly
labelled, so identity never rests on colour alone.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as C
import data as D
import rfblr as R

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.facecolor": SURFACE,
})
M_SUB = [2, 5, 9, 25, 50]


# ---------------------------------------------------------------- S1 curves
def s1_curves(n_draws=10, cache="results/s1_curves.npz"):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["post"], z["boot"], z["post_r"], z["boot_r"]
    post, boot, post_r, boot_r = [], [], [], []
    for i in range(n_draws):
        seed = C.SEED + i
        d = D.make_draw(np.random.default_rng(seed), 1.0)
        xy, obs = d["xy"], d["obs"]
        xy_obs, y_obs = xy[obs], d["y"][obs]
        s2 = d["noise_sd"][obs] ** 2
        phi = R.make_features(C.N_FEATURES, C.FOURIER_SCALE,
                              np.random.default_rng(seed + 10_000))
        Po, Pa = phi(xy_obs), phi(xy)
        mean, cov = R.fit_posterior(Po, y_obs, s2, C.PRIOR_PRECISION)
        _, epi = R.predict(Pa, mean, cov)
        den = np.maximum(epi, 1e-12)

        L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
        srng = np.random.default_rng(seed + 70_000)
        W = mean[None, :] + srng.standard_normal((max(M_SUB), cov.shape[0])) @ L.T
        mp = W @ Pa.T

        brng = np.random.default_rng(seed + 90_000)
        mb = []
        for _ in range(max(M_SUB)):
            ix = brng.integers(0, len(y_obs), len(y_obs))
            mj, _ = R.fit_posterior(Po[ix], y_obs[ix], s2[ix], C.PRIOR_PRECISION)
            mb.append(Pa @ mj)
        mb = np.array(mb)

        post.append([np.median(np.abs(mp[:M].var(0, ddof=1) - epi) / den) for M in M_SUB])
        boot.append([np.median(np.abs(mb[:M].var(0, ddof=1) - epi) / den) for M in M_SUB])
        post_r.append([np.median(mp[:M].var(0, ddof=1) / den) for M in M_SUB])
        boot_r.append([np.median(mb[:M].var(0, ddof=1) / den) for M in M_SUB])
    out = [np.array(a).T for a in (post, boot, post_r, boot_r)]
    np.savez(cache, post=out[0], boot=out[1], post_r=out[2], boot_r=out[3])
    return out


# ---------------------------------------------------------------- figure 1
def fig_scene():
    d = D.make_draw(np.random.default_rng(C.SEED), 1.0)
    g = C.GRID
    xy, obs = d["xy"], d["obs"]
    phi = R.make_features(C.N_FEATURES, C.FOURIER_SCALE,
                          np.random.default_rng(C.SEED + 10_000))
    mean, cov = R.fit_posterior(phi(xy[obs]), d["y"][obs],
                                d["noise_sd"][obs] ** 2, C.PRIOR_PRECISION)
    _, epi = R.predict(phi(xy), mean, cov)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    panels = [("Signal", d["truth"].reshape(g, g), cmap),
              ("True noise sd (aleatoric)", d["noise_sd"].reshape(g, g), cmap),
              ("Observed pixels", obs.reshape(g, g).astype(float), "gray_r"),
              ("Exact epistemic sd", np.sqrt(epi).reshape(g, g), cmap)]
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7))
    for ax, (title, img, cm) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cm, interpolation="nearest")
        ax.set_title(title, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if title != "Observed pixels":
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=INK2)
    r0, r1, c0, c1 = C.HOLE          # rows map to y, cols to x
    for ax in axes:
        ax.add_patch(plt.Rectangle((c0 * g, r0 * g), (c1 - c0) * g, (r1 - r0) * g,
                                   fill=False, ec=ORANGE, lw=1.4, ls=(0, (3, 2))))
    axes[2].annotate("unobserved block", (c1 * g, (r0 + r1) / 2 * g),
                     xytext=(14, 10), textcoords="offset points", color=ORANGE,
                     fontsize=7.5, fontweight="bold", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    tr0, tr1, tc0, tc1 = C.THIN_REGION
    axes[2].add_patch(plt.Rectangle((tc0 * g, tr0 * g), (tc1 - tc0) * g,
                                    (tr1 - tr0) * g, fill=False, ec=AQUA, lw=1.2,
                                    ls=(0, (2, 2))))
    axes[2].annotate(f"sampled at {C.THIN_FACTOR:g}x density",
                     ((tc0 + tc1) / 2 * g, tr0 * g), xytext=(0, -14),
                     textcoords="offset points", color=AQUA, fontsize=7.5,
                     fontweight="bold", ha="center", annotation_clip=False)
    fig.tight_layout()
    fig.savefig("figures/fig1_scene.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_convergence(z, m_values):
    post, boot, post_r, boot_r = s1_curves()
    epi_exact = z["base__epi_exact"]
    feat, feat_r = [], []
    for M in m_values:
        hat = z[f"base__epi_hat_M{M}"]
        feat.append(np.median(np.abs(hat - epi_exact)
                              / np.maximum(epi_exact, 1e-12), axis=1))
        feat_r.append(np.median(hat / np.maximum(epi_exact, 1e-12), axis=1))
    feat, feat_r = np.array(feat), np.array(feat_r)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.5))

    series = [("posterior samples", M_SUB, np.median(post, 1), BLUE),
              ("feature draws", m_values, np.median(feat, 1), ORANGE),
              ("data resamples", M_SUB, np.median(boot, 1), AQUA)]
    for name, xs, ys, col in series:
        a1.plot(xs, ys, "-o", color=col, lw=2, ms=4.5, mec=SURFACE, mew=0.8)
        a1.annotate(name, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    color=col, fontsize=8, va="center", fontweight="bold")
    ref = np.median(post, 1)[0] * np.sqrt((M_SUB[0] - 1) / (np.array(M_SUB, float) - 1))
    a1.plot(M_SUB, ref, ls=(0, (3, 2)), color=MUTED, lw=1.2)
    a1.annotate(r"$\sqrt{2/(M-1)}$", (M_SUB[-1], ref[-1]), xytext=(6, 6),
                textcoords="offset points", color=MUTED, fontsize=7.5)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("ensemble size M")
    a1.set_ylabel("median relative error of epistemic estimate")
    a1.set_title("Only true posterior samples converge")
    a1.set_xlim(1.7, 150)

    for name, xs, ys, col in [("posterior samples", M_SUB, np.median(post_r, 1), BLUE),
                              ("feature draws", m_values, np.median(feat_r, 1), ORANGE),
                              ("data resamples", M_SUB, np.median(boot_r, 1), AQUA)]:
        a2.plot(xs, ys, "-o", color=col, lw=2, ms=4.5, mec=SURFACE, mew=0.8)
        dy = 8 if name == "posterior samples" else 0
        a2.annotate(f"{name} ({ys[-1]:.2f}x)", (xs[-1], ys[-1]), xytext=(4, dy),
                    textcoords="offset points", color=col, fontsize=8, va="center",
                    fontweight="bold")
    a2.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    a2.annotate("truth", (1.9, 1.0), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=7.5)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("ensemble size M")
    a2.set_ylabel("estimate / true epistemic variance")
    a2.set_title("Each construction converges to a different quantity")
    a2.set_xlim(1.7, 300)
    fig.tight_layout()
    fig.savefig("figures/fig2_convergence.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_coverage(z):
    cg, ch = z["base__cov_conf_global"], z["base__cov_conf_hole"]
    rg, rh = z["base__cov_raw_global"], z["base__cov_raw_hole"]
    M = 9 if "base__mu_hat_M9" in z.files else max(list(z["m_values"]))
    mu, epi, alea = z[f"base__mu_hat_M{M}"], z["base__epi_exact"], z["base__alea_exact"]

    err_in, sd_in = [], []
    for r in range(len(ch)):
        d = D.make_draw(np.random.default_rng(C.SEED + r), 1.0)
        h = d["hole"]
        err_in.append(np.abs(mu[r][h] - d["y"][h]).mean())
        sd_in.append(np.sqrt(epi[r][h] + alea[r][h]).mean())
    err_in, sd_in = np.array(err_in), np.array(sd_in)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.5))

    groups = [("raw\nglobal", rg, MUTED), ("conformal\nglobal", cg, BLUE),
              ("raw\nin hole", rh, MUTED), ("conformal\nin hole", ch, ORANGE)]
    rng = np.random.default_rng(3)
    for i, (name, arr, col) in enumerate(groups):
        a1.scatter(i + rng.uniform(-0.13, 0.13, len(arr)), arr, s=18, color=col,
                   alpha=0.65, lw=0.6, edgecolor=SURFACE, zorder=3)
        a1.plot([i - 0.28, i + 0.28], [arr.mean()] * 2, color=INK, lw=2, zorder=4)
    a1.axhline(C.NOMINAL_COVERAGE, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    a1.annotate("nominal 90%", (-0.42, C.NOMINAL_COVERAGE), xytext=(0, 4),
                textcoords="offset points", color=MUTED, fontsize=7.5, ha="left")
    a1.set_xticks(range(4)); a1.set_xticklabels([g[0] for g in groups])
    a1.set_ylim(0, 1.02); a1.set_ylabel("empirical coverage")
    a1.set_title("Marginal coverage is exact; regional coverage scatters")
    a1.annotate("one point per draw", (0, 0.06), color=MUTED, fontsize=7.5)

    a2.scatter(err_in, ch, s=26, color=ORANGE, alpha=0.8, lw=0.6, edgecolor=SURFACE,
               zorder=3, label="actual error in hole")
    a2.scatter(sd_in, ch, s=26, color=BLUE, alpha=0.8, lw=0.6, edgecolor=SURFACE,
               zorder=3, label="predicted sd in hole")
    a2.annotate(f"actual error\nr = {np.corrcoef(err_in, ch)[0,1]:+.2f}",
                (err_in.max(), ch[np.argmax(err_in)]), xytext=(-6, 14),
                textcoords="offset points", color=ORANGE, fontsize=8,
                fontweight="bold", ha="right")
    a2.annotate(f"predicted sd\nr = {np.corrcoef(sd_in, ch)[0,1]:+.2f}",
                (sd_in.mean(), 0.30), xytext=(26, 0), textcoords="offset points",
                color=BLUE, fontsize=8, fontweight="bold", ha="left", va="center")
    a2.axhline(C.NOMINAL_COVERAGE, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    a2.set_xlabel("mean magnitude inside the unobserved block")
    a2.set_ylabel("conformal coverage in hole")
    a2.set_title("The estimate does not track the error it describes")
    a2.set_ylim(0.15, 1.05)
    fig.tight_layout()
    fig.savefig("figures/fig3_coverage.png", dpi=200)
    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)
    z = np.load("results/raw.npz")
    m_values = list(z["m_values"])
    fig_scene()
    fig_convergence(z, m_values)
    fig_coverage(z)
    for f in ("fig1_scene", "fig2_convergence", "fig3_coverage"):
        print(f"figures/{f}.png", os.path.getsize(f"figures/{f}.png"), "bytes")


if __name__ == "__main__":
    main()
