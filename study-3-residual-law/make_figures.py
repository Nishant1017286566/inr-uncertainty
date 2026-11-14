"""Figures. Three, in the order the argument runs:
  fig1_anomaly      - A moves 679x while sigma^2 is fixed: the three sweeps as they look now
  fig2_collapse     - the same 97 settings on one line, with the interventions on top
  fig3_consequence  - what it costs: ranking quality by regime, and the confidently-wrong case
Palette: validated for colour-vision-deficient separation; every series also directly labelled.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8d5bd6"
INK, INK2, MUTED, SURFACE = "#14130f", "#44433d", "#817f76", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": .8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.facecolor": SURFACE,
})


def grid(v="results_images_v3"):
    d = json.load(open(f"grids/{v}.json"))
    k = sorted(d)
    g = lambda f: np.array([d[s][f] for s in k], float)
    return dict(A=g("A"), E=g("E"), err=g("err"), sig=g("sigma"), tr=g("train_rmse"),
                n=g("n"), w=g("width"), ns=g("noise"), rho=g("rho_std_err"))


def sweep(d, knob, base=dict(n=10, width=50, noise=10)):
    """The one-at-a-time sweep: settings where the other two knobs sit at the baseline."""
    others = [k for k in ("n", "width", "noise") if k != knob]
    key = {"n": "n", "width": "w", "noise": "ns"}
    m = np.ones(len(d["A"]), bool)
    for o in others:
        m &= d[key[o]] == base[o]
    i = np.argsort(d[key[knob]][m])
    return {f: d[f][m][i] for f in d}


# ------------------------------------------------------------------ figure 1
def fig1():
    d = grid()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, (knob, label) in zip(axes, [("n", "images in one network"),
                                        ("width", "hidden units"),
                                        ("noise", "noise (% of image std)")]):
        s = sweep(d, knob)
        x = s[{"n": "n", "width": "w", "noise": "ns"}[knob]]
        ax.plot(x, s["A"], "-o", color=ORANGE, lw=2, ms=4.5, mec=SURFACE, mew=.8)
        ax.plot(x, s["sig"] ** 2, "--", color=INK, lw=1.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_title(label.split("(")[0].strip())
    axes[0].set_ylabel("variance")
    axes[0].annotate("A, the reported\n'aleatoric' variance", (2, 6e-5),
                     color=ORANGE, fontsize=8, fontweight="bold")
    axes[0].annotate("true noise variance", (2, 2.4e-4), color=INK, fontsize=8,
                     fontweight="bold")
    axes[2].annotate("only here does the\ntruth move at all", (3, 3e-6), color=INK,
                     fontsize=7.5)
    fig.suptitle("The quantity A is meant to estimate is flat in two of three sweeps. "
                 "A is not.", y=1.02, fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig("figures/fig1_anomaly.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig2():
    d = grid()
    I = json.load(open("results/interventions_both.json"))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    x, y = d["tr"] / d["sig"], d["A"] / d["sig"] ** 2
    knobmark = [("n", "n", "o"), ("width", "w", "s"), ("noise", "ns", "^")]
    for knob, key, mk in knobmark:
        s = sweep(d, knob)
        ax.scatter(s["tr"] / s["sig"], s["A"] / s["sig"] ** 2, s=34, marker=mk,
                   color=BLUE, alpha=.85, lw=.6, edgecolor=SURFACE, zorder=3)
    rest = np.ones(len(x), bool)
    for knob, key, mk in knobmark:
        s = sweep(d, knob)
        for t, a in zip(s["tr"], s["A"]):
            rest &= ~((d["tr"] == t) & (d["A"] == a))
    ax.scatter(x[rest], y[rest], s=16, marker="o", color=BLUE, alpha=.30, lw=0, zorder=2)

    sl, ic = np.polyfit(np.log(x), np.log(y), 1)
    xx = np.logspace(np.log10(x.min() * .9), np.log10(x.max() * 1.1), 50)
    ax.plot(xx, np.exp(ic) * xx ** sl, color=INK, lw=1.5, zorder=4)
    ax.plot(xx, xx ** 2, ls=(0, (3, 2)), color=MUTED, lw=1.2, zorder=4)

    for tag, col, mk, lbl in (("I2", ORANGE, "D", "training budget (I2)"),
                              ("I1", AQUA, "P", "task difficulty (I1)")):
        R = [r for r in I[tag] if np.isfinite(r["A"])]
        tr = np.array([r["train_rmse"] / r["sigma"] for r in R])
        a = np.array([r["A"] / r["sigma"] ** 2 for r in R])
        ax.scatter(tr, a, s=46, marker=mk, color=col, alpha=.95, lw=.7,
                   edgecolor=SURFACE, zorder=5, label=lbl)

    ax.axhline(1, color=MUTED, lw=.9, ls=":")
    ax.axvline(1, color=MUTED, lw=.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("train_rmse / $\\sigma$   (where the fit sits relative to the noise floor)")
    ax.set_ylabel("A / $\\sigma^2$   (how wrong the noise estimate is)")
    ax.annotate(f"fitted slope {sl:.2f}", (x.max() * .30, np.exp(ic) * (x.max() * .30) ** sl),
                xytext=(6, -16), textcoords="offset points", color=INK, fontsize=8.5,
                fontweight="bold", ha="left")
    ax.annotate("slope 2 (NLL optimum)", (xx[-14], xx[-14] ** 2), xytext=(-6, 8),
                textcoords="offset points", color=MUTED, fontsize=7.5, ha="right")
    ax.annotate("A correct", (x.min() * 1.02, 1), xytext=(0, 5),
                textcoords="offset points", color=MUTED, fontsize=7.5)
    ax.annotate("memorising", (0.55, 40), color=MUTED, fontsize=7.5, ha="center")
    ax.annotate("underfitting", (4.2, 40), color=MUTED, fontsize=7.5, ha="center")
    ax.scatter([], [], s=34, marker="o", color=BLUE, label="97-setting grid (3 sweeps)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("One law: where the fit sits relative to the noise floor predicts\n"
                 "the entire error in the aleatoric estimate", loc="left")
    fig.tight_layout()
    fig.savefig("figures/fig2_collapse.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig3():
    d = grid()
    I = json.load(open("results/interventions_both.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # left: A and E under I2 - the asymmetry
    R = [r for r in I["I2"] if np.isfinite(r["A"])]
    st = sorted(set(r["knob"] for r in R))
    for nm, f, col, mk in (("A  (aleatoric)", "A", ORANGE, "o"),
                           ("E  (epistemic)", "E", BLUE, "s")):
        m = [np.mean([r[f] for r in R if r["knob"] == s]) for s in st]
        a1.plot(st, m, "-", marker=mk, color=col, lw=2, ms=5, mec=SURFACE, mew=.8)
        a1.annotate(nm, (st[-1], m[-1]), xytext=(-6, 10 if f == "A" else -14),
                    textcoords="offset points", color=col, fontsize=8.5,
                    fontweight="bold", ha="right")
    sg = np.mean([r["sigma"] ** 2 for r in R])
    a1.axhline(sg, color=INK, ls="--", lw=1.3)
    a1.annotate("true noise variance (fixed)", (st[0], sg), xytext=(2, 6),
                textcoords="offset points", color=INK, fontsize=7.5, fontweight="bold")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xticks(st); a1.set_xticklabels([f"{int(v)}" for v in st])
    a1.minorticks_off()
    a1.set_xlabel("training steps (nothing else changes)")
    a1.set_ylabel("variance")
    a1.set_title("Train longer, and the 'noise in the data' falls 12.6x", loc="left")

    # right: does the uncertainty point at the errors? (rho_std_err, NOT the A-vs-E rho)
    ns_, rho = d["ns"], d["rho"]
    ok = np.isfinite(rho)
    x2 = d["tr"] / d["sig"]
    base = (d["n"] == 10) & (d["w"] == 50)
    a2.scatter(ns_[ok & ~base], rho[ok & ~base], s=18, color=BLUE, alpha=.35, lw=0,
               zorder=2)
    sw = base & ok
    i = np.argsort(ns_[sw])
    a2.plot(ns_[sw][i], rho[sw][i], "-o", color=ORANGE, lw=2, ms=5, mec=SURFACE, mew=.8,
            zorder=4)
    a2.annotate("the noise sweep", (ns_[sw][i][-3], rho[sw][i][-3]), xytext=(-6, 14),
                textcoords="offset points", color=ORANGE, fontsize=8, fontweight="bold",
                ha="right")
    a2.annotate("other 86 settings", (1.3, 0.80), color=MUTED, fontsize=7.5)
    a2.axhline(0, color=INK, lw=1.0)
    a2.annotate("never negative: the uncertainty degrades\nto useless, it does not invert",
                (1.25, 0.04), color=INK, fontsize=7.5)
    a2.set_xscale("log")
    a2.set_ylim(-0.08, 0.92)
    a2.set_xlabel("noise (% of image std)")
    a2.set_ylabel("rank corr(predicted sd, true |error|)")
    a2.set_title("Ranking quality collapses with noise (0.52 $\\rightarrow$ 0.03)",
                 loc="left")
    fig.tight_layout()
    fig.savefig("figures/fig3_consequence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig1(); fig2(); fig3()
    for f in ("fig1_anomaly", "fig2_collapse", "fig3_consequence"):
        print("figures/%s.png %d bytes" % (f, os.path.getsize(f"figures/{f}.png")))
