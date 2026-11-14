# Study 1 — the sweeps

The exploratory study that produced the data studies 2 and 3 explain. One implicit neural
representation, `f(x, y, image code) → (mean, log-variance)`, trained on `n` synthetic images
at once: each image gets a learned code vector, all images share the weights, and five
independent networks per setting are trained together as one batched model.

Three knobs, each from 1 to 100:

| knob | meaning |
|---|---|
| **n** | how many images the one network must represent |
| **width** | hidden units (3 hidden layers, Fourier-feature input) |
| **noise** | observation noise as a percentage of the images' own standard deviation |

Each image is observed through noisy pixels; 20% of observed pixels are held out for
conformal calibration; a block in the middle of every image is never observed. The clean
image is analytic, so the true value — and therefore the true error — is known at any
coordinate, and the noise level is set by the experimenter.

A full 100×100×100 grid is a million trainings, so the design is three one-at-a-time sweeps
plus three pairwise heatmaps: **97 distinct settings**.

## What it records, per setting

`A` (mean predicted variance, read as aleatoric), `E` (variance of predicted means across
the five networks, read as epistemic), the true error inside and outside the unobserved
block, `train_rmse`, rank correlations between predicted uncertainty and true error, raw and
conformal coverage at four nominal levels, and — in v3 — the correlation between `A` and `E`
across points with block-bootstrap intervals.

## What it showed, and what it could not settle

The sweeps showed `A` moving by orders of magnitude in settings where the true noise is
unchanged by construction, `E` rising with the noise level it is not supposed to track, and
rank correlation with the true error collapsing as noise grows. What they could not settle is
*why*: with five networks and no computable ground truth for the decomposition itself, a bad
estimator, too few ensemble members, and a decomposition that does not do what it claims all
predict the same observations.

[Study 2](../study-2-exact-posterior) removes that ambiguity by moving to a model with a
closed-form posterior. [Study 3](../study-3-residual-law) comes back and explains these grids
with a single scaling law.

## Contents

| | |
|---|---|
| `inr_images_n_width_noise_v3.py` | the study as a script (`--quick`, `--estimate-only`, `--shard i/k`) |
| `notebooks/` | the same study as notebooks, v1 → v3, with outputs |
| `grids/all_settings_v{1,2,3}.json` | **the completed 97-setting grids**, one per version |

The three versions differ in what they record, not in the model: v2 adds the calibration
block, v3 adds the `A`–`E` correlation analysis. The model, seeds and image family are
unchanged, which is why the law in study 3 fits all three with the same exponent.

The grids are what studies 2 and 3 read, so neither needs this re-run. Reproducing the grids
themselves wants a GPU — the full set took about 28 minutes on a laptop RTX 3070 Ti, and the
script prints a projected run time for the current machine before starting anything long.

```bash
pip install -r requirements.txt
python3 inr_images_n_width_noise_v3.py --estimate-only   # time projection only
python3 inr_images_n_width_noise_v3.py                   # the full grid, cached per setting
```

## Known issue

The image family is well defined only for `DIVERSITY < 3.33`. Above that, the blob-width term
`0.020·(1 + D·U(−0.3, 0.3))` can go negative, flipping a Gaussian into a divergent
exponential and producing non-finite pixels — at `D = 4`, for 8 of the 100 images. The grids
here were all produced at the default `D = 1` and are unaffected. See study 3, which hit it.
