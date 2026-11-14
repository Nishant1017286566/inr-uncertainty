# Uncertainty decomposition in implicit neural representations

**Three studies asking what the aleatoric–epistemic split actually measures. The answer is
not what it is named after.**

Predictive uncertainty is conventionally split in two: an *aleatoric* part, irreducible
noise in the data, and an *epistemic* part, the model's own ignorance that more data should
reduce. The split matters because it implies opposite actions — aleatoric means measure
better, epistemic means measure more. These studies test whether the two numbers support
that decision on a task where every cause of uncertainty is known by construction.

They do not.

| | finding | strength of evidence |
|---|---|---|
| The **aleatoric** head reports the **training residual**, not the noise | one scaling law, exponent ≈ 2, collapses 97 settings spanning 679× | observational grid + 2 pre-registered interventions |
| The **epistemic** term converges on the wrong quantity | ensemble spread → 0.44× (initialisation) or 6.72× (bootstrap) of the true value, never 1.00× | exact closed-form posterior |
| Both parts are badly wrong; **their sum is nearly right** | 0.44× and 1.43× → total **1.10×** | exact closed-form posterior |
| Calibration fixes the **marginal** number and nothing regional | conformal: 0.900 global, 0.24–0.97 inside an unobserved block | 30 pre-registered draws |

The practical consequence, in one line: **report the total, and report `train_rmse/σ`
beside any aleatoric number — or do not report one.**

## Paper

A 9-page write-up of all three studies is in [`paper/`](paper), with the compiled
[preprint PDF](paper/residual-law-preprint.pdf), the LaTeX source, and the fields to paste
into an arXiv submission. Every statistic in it is reproduced by the scripts in this
repository.

---

## The three studies, in the order they should be read

### [Study 1 — the sweeps](study-1-sweeps)
One INR representing `n` images at once, with three knobs from 1 to 100: number of images,
network width, and observation noise. 97 settings, five networks each, every clean value
known analytically. This is the exploratory study that produced the data the others explain —
three sweeps and three pairwise heatmaps, run across three successive versions of the code.

Its own notebooks and the completed `all_settings.json` for all three versions are here, so
every number the later studies attribute to it can be recomputed.

### [Study 2 — the exact posterior](study-2-exact-posterior)
Study 1 can show that the decomposition misbehaves but not *why*, because with an ensemble of
five networks there is no way to separate a bad estimator, too few members, and a
decomposition that does not do what it claims.

So: replace the trained layers above the INR's fixed random Fourier features with a Bayesian
linear head. The result — random-feature Bayesian linear regression — is a close analogue
with a posterior in closed form, where the true aleatoric and epistemic terms are *computable*
rather than estimated. Four pre-registered hypotheses, three rejected.

The headline: an ensemble's spread estimates the variance of whatever distinguishes its
members, and that equals the posterior's epistemic variance only when the members *are*
posterior samples. They never are.

### [Study 3 — the residual law](study-3-residual-law)
Back to the real INR, with the question study 2 could not ask of it: what is the aleatoric
head doing?

The Gaussian NLL term `0.5·(log v + r²/v)` is minimised at `v = r²` — the squared residual on
the *training* data. So the head reports that residual whatever produced it, and equals the
noise variance only where the fit happens to sit on the noise floor. All 97 settings of study
1 collapse onto that one law with slope 1.86–1.92 (theory: 2) and R² 0.93–0.96, in every
version of the code.

Two pre-registered interventions make it causal. The sharpest: **train the same model on the
same data 27× longer and the reported "noise in the data" falls 12.6×**, while the true noise
is fixed by construction.

---

## How to read the evidence

Each study states its hypotheses and decision rules before its results, in a
`preregistration.md`, and reports failures at the same length as successes. Three of four
pre-registered hypotheses failed in study 2. Study 3's falsifier failed and is rescued only
by a post-hoc split, which is said plainly rather than quietly rewritten.

One claim in study 2 is attributed to an earlier exploratory run whose code was not
recoverable. It is marked as such, is not offered as evidence, and is not counted among the
findings above. Everything else in this repository is reproduced by the scripts beside it.

A latent bug found along the way, in study 1's own image generator, is documented in study 3:
at `DIVERSITY ≥ 3.33` the blob-width term goes negative and flips a Gaussian into a divergent
exponential, silently producing non-finite pixels. It is documented rather than patched,
because the generated images are what the completed grids were run on.

---

## What this is not

Synthetic data throughout, two model families, one noise model, and that noise is Gaussian
and heteroscedastic exactly as the models assume — real misspecification would likely make
the picture worse, not better. Read these as careful negative results on a controlled test
bed, not as general claims about uncertainty quantification.

The reason for the care is the subject matter. If a decomposition cannot beat the
undecomposed quantity at the decision it exists to inform, the interpretability it adds is
not yet evidence. Validating an attribution turns out to be harder than producing one, which
is the actual lesson of all three studies.

---

## Running them

Each folder is self-contained, with its own `requirements.txt` and reproduction commands.
Studies 2 and 3 run on CPU in about 15 and 21 minutes respectively; study 1 wants a GPU for
the full grid, and its completed grids are included so the other studies need no re-run.

```bash
cd study-3-residual-law && pip install -r requirements.txt
python3 analyse.py && python3 make_figures.py     # regenerates every number and figure
```

---

**Nishant Dave** — MSc Data & Artificial Intelligence, CentraleSupélec / Université
Paris-Saclay; B.Tech, VIT. Working on speech and audio machine learning, with a particular
interest in whether the evaluations we trust actually measure what they claim.

These studies are self-directed and unfunded, run on a laptop GPU and a CPU box. MIT
licensed — reuse, rerun and contradict them freely. Corrections are welcome and will be
recorded rather than quietly folded in.
