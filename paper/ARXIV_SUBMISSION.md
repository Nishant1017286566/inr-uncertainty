# arXiv submission — fields to paste

Upload `arxiv-submission.tar.gz` at <https://arxiv.org/submit>. It contains `main.tex` and
four PNGs and compiles standalone to 9 pages with zero undefined references (verified from
the tarball alone, in a clean directory). Do **not** upload the PDF — arXiv builds it.

---

## Title

```
What the Aleatoric Head Measures: A Residual Law for Uncertainty Decomposition in Implicit Neural Representations
```

## Authors

```
Nishant Dave
```

## Abstract (plain text — arXiv strips LaTeX from this box, so this version has none)

```
Predictive uncertainty is routinely split into an aleatoric part, meant to capture noise in
the data, and an epistemic part, meant to capture the model's own ignorance. We ask what
these two numbers measure in practice, on a task where every cause of uncertainty is fixed
by construction. Training one implicit neural representation over n images with a Gaussian
negative-log-likelihood head and a five-member ensemble, across a 97-setting grid over data
volume, capacity and noise, we find that the reported aleatoric variance A moves by 679x
while the true noise variance it estimates is held constant over most of the grid, and lands
within 1.5x of that truth in only 13% of settings. One relation accounts for this: A/sigma^2
is a power law in RMSE_train/sigma with fitted exponent 1.86-1.92 (R^2 = 0.93-0.96), which is
the optimum of the loss the head is trained on. The aleatoric head is a meter for the
training residual, and coincides with the noise only where the fit happens to sit on the
noise floor. Two pre-registered interventions establish this causally: raising task
difficulty at fixed capacity, data and noise moves A by 6.4x (rho = +0.98), and training the
same model on the same data 27x longer - changing nothing about the data, model or noise -
cuts the reported "noise in the data" by 12.6x (rho = -0.98). In a companion analysis on a
model whose posterior is available in closed form, we show the epistemic term is no better
off: an ensemble's spread converges to 0.44x the true epistemic variance when members differ
by initialisation and 6.72x when they differ by resampling, never to 1.00x, while the two
biases partly cancel so that the undecomposed total is accurate to 1.10x. We report one
pre-registered falsifier that failed. We recommend reporting total predictive variance, and
reporting RMSE_train/sigma beside any aleatoric number.
```

## Primary category

```
cs.LG  (Machine Learning)
```

## Cross-list

```
stat.ML  (Machine Learning, Statistics)
```

## Comments

```
9 pages, 3 figures. Pre-registrations, code, completed grids and scripts that regenerate
every number and figure: https://github.com/Nishant1017286566/inr-uncertainty
```

## License

Recommended: **CC BY 4.0**. It permits reuse with attribution, which is what you want for
work whose purpose is to be checked and cited. The arXiv default (non-exclusive licence to
distribute) is more restrictive than it needs to be here.

---

## One thing to expect: endorsement

If this is your first arXiv submission to `cs.LG`, arXiv may ask for an **endorsement** from
an established submitter in that archive before it will accept it. This is routine and not a
quality judgement — it is arXiv's spam control.

If you are asked:

- Any researcher who has submitted to `cs.LG` recently can endorse. Supervisors, lab members,
  or co-authors on earlier work are the usual route.
- arXiv gives you an endorsement code and a link to pass on. The endorser does not review the
  paper; they attest that you are a researcher working in the area.
- A CentraleSupélec or Paris-Saclay contact is the natural first ask. Nora Ouzir or Francesca
  Bugiotti are both plausible, and both already know your work from the referee requests.

If no endorsement is forthcoming, the fallback that costs nothing is Zenodo: it issues a DOI
immediately with no gatekeeping, and the paper can go to arXiv later as well. The GitHub
repository is citable in the meantime.

---

## Before you submit — two judgement calls that are yours

1. **Affiliation.** The paper currently carries your name and a footnote saying the work was
   carried out independently of any institution. That is accurate. If you would rather list
   CentraleSupélec / Université Paris-Saclay, say so and I will change it — but only if the
   work genuinely sits under that affiliation, since claiming an institution for self-directed
   work is the kind of thing that is awkward to walk back.

2. **The failed falsifier.** Section 6 reports in full that a pre-registered hypothesis failed
   and was partially rescued post-hoc. A reviewer will notice it. Leaving it in is the honest
   choice and, in our view, strengthens the paper — but it is a choice, and you should make it
   knowingly rather than discover it at an interview.
