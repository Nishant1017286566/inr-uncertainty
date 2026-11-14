# %% [markdown]
# # How does INR uncertainty change with the NUMBER OF IMAGES, the network size and the noise?
#
# One implicit neural representation `f(x, y, image id) -> (mean, log-variance)` is trained on **N images at once** (each image gets
# a learned code vector; all images share the weights). Each image is a different synthetic picture from one family (blobs, a disc,
# a stripe pattern, random positions and strengths) and is observed through 30% of its pixels with Gaussian noise. Three knobs,
# each from **1 to 100**:
#
# | Knob | Meaning of 1 ... 100 |
# |---|---|
# | **n**: number of images | how many images the one network must represent (images 0..n-1; image 0 is always included) |
# | **width**: network size | hidden-layer width, 1 to 100 units (3 hidden layers, Fourier-feature input) |
# | **noise**: observation noise | noise standard deviation as a percent of the images' own standard deviation |
#
# **Uncertainty measures** (M seeds per setting): `A` = mean predicted variance (aleatoric: should track the noise variance),
# `E` = variance of the predicted means across seeds (epistemic: model disagreement). The truth is known, so we also record the
# **true error** and whether the uncertainty tracks it. Uncertainty is evaluated at random off-grid points of up to 10 of the images
# (`E`, `err`, ... = average over those images); `E0`, `err0` are for **image 0 only**, which is in every run, so it is the strictly
# comparable curve as n grows. A block in the middle of every image never has training pixels.
#
# **Calibration.** Each image also has 20% of its observed pixels held out (noisy, never trained on). They are used to *check* whether
# the predicted uncertainty is honest (does a 90% interval contain the truth 90% of the time?) and to *fix* it with split-conformal
# scaling. We then test on fresh noisy values at random points, outside the empty block and inside it (where calibration should break).
#
# **Speed.** The M networks of one setting are trained together as ONE batched model (independent weights, `torch.baddbmm`), which
# cuts the number of GPU kernel launches about M-fold. These networks are tiny, so launch overhead, not arithmetic, dominates.
#
# **GPU.** If PyTorch can see an NVIDIA GPU the notebook uses it automatically (the device is printed in the settings cell). On Windows a plain
# `pip install torch` often installs the CPU-only build; for the GPU build run once in a terminal:
# `pip install torch --index-url https://download.pytorch.org/whl/cu124` (or pick your CUDA version at pytorch.org/get-started).
#
# **Design.** A full 100 x 100 x 100 grid is a million trainings, so we do three **one-at-a-time sweeps** and three **pairwise
# heatmaps**. The cell after the settings **benchmarks your machine and prints the projected run time** before anything long starts.

# %%
import sys, os, math, time, json, pickle, subprocess, warnings
warnings.filterwarnings("ignore")
try:
    import torch
except ImportError:                                   # e.g. a fresh Colab / venv
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Settings (edit here)

# %%
QUICK = False        # in a notebook, just set QUICK = True/False by hand
G = 32                               # each image is G x G pixels
PIX_FRAC = 1.0                       # fraction of each image's (usable) pixels that is observed
CAL_FRAC = 0.2                       # of those, the fraction held out for calibration (never trained on)
M = 5                                # ensemble size (networks) per setting
STEPS = 1000  # overridden by I2                         # optimisation steps per model
K_SWEEP = 12                         # values per one-at-a-time sweep (use 100 to try EVERY integer 1..100)
K_PAIR = 5                           # values per axis in the pairwise heatmaps
BASE = dict(n=10, width=50, noise=10)    # the held-fixed values when another knob is varied
N_EVAL = 10                          # evaluate on (up to) this many of the images
DIVERSITY = 1.0                      # 0 = all images identical, 1 = default variety, 2 = very different images
N_QUERY = 3000                       # random query points per image
LEVELS = (0.5, 0.8, 0.9, 0.95)       # nominal interval levels for calibration
EMB = 8                              # size of the learned per-image code
NFEAT, SCALE, WD, LAYERS = 32, 1.5, 1e-3, 3
SEED = 0
OUT = "results_images_v3"
NTHREADS = int(os.environ.get("NTHREADS", os.cpu_count() or 1))
SHARD = None                         # command-line only: --shard i/k runs every k-th job (for parallel runs)
if "--shard" in sys.argv:
    SHARD = tuple(int(v) for v in sys.argv[sys.argv.index("--shard") + 1].split("/"))
if QUICK:
    G, M, STEPS, K_SWEEP, K_PAIR, N_QUERY, OUT = 20, 3, 300, 4, 3, 600, "results_images_v3_quick"
os.makedirs(os.path.join(OUT, "cells"), exist_ok=True)
torch.set_num_threads(NTHREADS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # uses your NVIDIA GPU automatically if PyTorch can see it
print(f"G={G} M={M} steps={STEPS} K_SWEEP={K_SWEEP} K_PAIR={K_PAIR} diversity={DIVERSITY} threads={NTHREADS} quick={QUICK} device={DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

# %% [markdown]
# ## 2. The world: 100 synthetic images with known clean values everywhere

# %%
from scipy.stats import norm
NMAX = 100


def image_fn(X, k):
    """Clean image number k on [0,1]^2 (analytic, so it can be evaluated at any coordinate). Image k is always the same."""
    r, D = np.random.default_rng(100 + k), DIVERSITY
    j = lambda s: D * r.uniform(-s, s)
    x0, x1 = X[:, 0], X[:, 1]
    f = 0.35 + (0.25 + j(0.15)) * x0 - (0.10 + j(0.15)) * x1
    f = f + (0.30 + j(0.10)) * np.exp(-(((x0 - 0.30 - j(0.12)) ** 2 + (x1 - 0.35 - j(0.12)) ** 2) / (0.020 * (1 + j(0.3)))))
    f = f + (0.25 + j(0.10)) * (((x0 - 0.70 - j(0.10)) ** 2 + (x1 - 0.65 - j(0.10)) ** 2) < 0.035)
    f = f + 0.08 * np.sin(2 * np.pi * (5 + round(j(1.5))) * x0 + j(1.5)) * np.sin(2 * np.pi * (4 + round(j(1.5))) * x1 + j(1.5))
    return f.astype(np.float32)


def in_hole(X):
    """A block in the middle that never has training or calibration samples (in every image)."""
    return (np.abs(X[:, 0] - 0.5) < 0.12) & (np.abs(X[:, 1] - 0.5) < 0.12)


c = (np.arange(G) + 0.5) / G
GRID = np.stack(np.meshgrid(c, c, indexing="xy"), -1).reshape(-1, 2).astype(np.float32)
OK = np.where(~in_hole(GRID))[0]
N_OBS = int(round(PIX_FRAC * len(OK)))
NPER = int(round((1 - CAL_FRAC) * N_OBS))                        # training pixels per image
CLEAN = np.stack([image_fn(GRID, k) for k in range(NMAX)])       # (100, G*G)
STD_CLEAN = float(CLEAN.std(axis=1).mean())                      # 100% noise = as large as a typical image's own std
_obs = [np.random.default_rng(SEED + 10 + k).permutation(OK)[:N_OBS] for k in range(NMAX)]
OBS = [o[:NPER] for o in _obs]                                   # training pixels of image k (fixed, nested in n)
CAL = [o[NPER:] for o in _obs]                                   # held-out calibration pixels of image k
EPS = np.random.default_rng(SEED + 1).normal(size=CLEAN.shape).astype(np.float32)           # one fixed noise draw per pixel
XQ = np.random.default_rng(SEED + 2).random((N_QUERY, 2)).astype(np.float32)
HQ = in_hole(XQ)
FQ = np.stack([image_fn(XQ, k) for k in range(NMAX)])
EPSQ = np.random.default_rng(SEED + 3).normal(size=FQ.shape).astype(np.float32)             # fresh noise for the test targets
print(f"{NMAX} images, {len(OK)} usable pixels each: {NPER} train + {len(CAL[0])} calibration, image std {STD_CLEAN:.3f}")


def n_points(n): return n * NPER

# %% [markdown]
# ## 3. The INR (M networks trained together as one batch) and one training run

# %%
class EnsINR(nn.Module):
    """M independent Fourier-feature MLPs (+ a learned code per image) evaluated at once with batched matrix products."""

    def __init__(self, width, n_img, M):
        super().__init__()
        g = torch.Generator().manual_seed(1234)                                       # CPU generator: same weights on any device
        self.register_buffer("B", torch.randn(M, 2, NFEAT, generator=g) * SCALE)
        self.emb = nn.Parameter(torch.randn(M, n_img, EMB, generator=g) * 0.3)
        dims = [2 * NFEAT + EMB] + [width] * LAYERS + [2]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for din, dout in zip(dims[:-1], dims[1:]):
            lim = 1 / math.sqrt(din)
            self.W.append(nn.Parameter((torch.rand(M, din, dout, generator=g) * 2 - 1) * lim))
            self.b.append(nn.Parameter((torch.rand(M, 1, dout, generator=g) * 2 - 1) * lim))

    def forward(self, x, img):                                                        # x: (P,2), img: (P,)  ->  (M,P), (M,P)
        z = 2 * math.pi * torch.einsum("pd,mdf->mpf", x, self.B)
        h = torch.cat([torch.sin(z), torch.cos(z), self.emb[:, img]], -1)
        for i in range(len(self.W)):
            h = torch.baddbmm(self.b[i], h, self.W[i])
            if i < len(self.W) - 1: h = torch.relu(h)
        return h[..., 0], -10.0 + 10.0 * torch.sigmoid(h[..., 1])                     # mean, log-variance in (-10, 0)


def fit(width, n_img, xt, it, yt):
    m = EnsINR(width, n_img, M).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=WD, fused=(DEVICE.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    warm = int(0.25 * STEPS)
    for t in range(STEPS):
        mu, lv = m(xt, it)
        per = ((mu - yt) ** 2).mean(1) if t < warm else (0.5 * (lv + (mu - yt) ** 2 * torch.exp(-lv))).mean(1)
        opt.zero_grad(set_to_none=True); per.sum().backward(); opt.step(); sched.step()      # sum over members = independent training
    return m


def make_data(n, sigma):
    xs = np.concatenate([GRID[OBS[k]] for k in range(n)])
    ids = np.concatenate([np.full(NPER, k) for k in range(n)])
    ys = np.concatenate([CLEAN[k][OBS[k]] + sigma * EPS[k][OBS[k]] for k in range(n)])
    return torch.tensor(xs).to(DEVICE), torch.tensor(ids).to(DEVICE), torch.tensor(ys.astype(np.float32)).to(DEVICE)

# %% [markdown]
# ## 4. One setting (n images, width, noise): train, split the uncertainty, compare with the truth, check the calibration
# Calibration, in words: on the held-out calibration pixels we compute `|y - mean| / predicted_std`. If the predicted std were honest,
# a fraction L of these scores would fall below the normal quantile z_L. **Conformal scaling** replaces z_L by the L-quantile of the
# scores actually seen, so the corrected intervals `mean +- q_L * std` get the right coverage on new data from the same distribution.

# %%
def rho(a, b, mask):
    a, b = a[mask], b[mask]
    return float(spearmanr(a, b)[0]) if len(a) > 5 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


N_BOOT, N_BLOCK = 200, 6                                          # bootstrap resamples; the unit square is cut into 6 x 6 spatial blocks


def ae_correlation(A, E, X, seed=0):
    """Correlation between aleatoric A and epistemic E across points, with a block-bootstrap 95% interval.
    A, E: (images, points) outside the hidden block; X: (points, 2) their coordinates.
    rho_log = Pearson on (log A, log E); rho_rank = Spearman (Pearson on global ranks); share = median of E / (A + E)."""
    from scipy.stats import rankdata
    ne = A.shape[0]
    a, e = np.log(np.maximum(A.ravel(), 1e-30)), np.log(np.maximum(E.ravel(), 1e-30))
    ra, re_ = rankdata(a), rankdata(e)
    share = (E / (A + E)).ravel()
    blk = np.tile((np.floor(X[:, 0] * N_BLOCK) * N_BLOCK + np.floor(X[:, 1] * N_BLOCK)).astype(int), ne)
    idx = [np.where(blk == b)[0] for b in range(N_BLOCK ** 2)]; idx = [i for i in idx if len(i)]
    pear = lambda u, v, i: float(np.corrcoef(u[i], v[i])[0, 1])
    allidx = np.arange(len(a))
    est = dict(rho_log=pear(a, e, allidx), rho_rank=pear(ra, re_, allidx), share=float(np.median(share)))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        bs.append((pear(a, e, pick), pear(ra, re_, pick), float(np.median(share[pick]))))
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    return dict(rho_log=est["rho_log"], rho_log_lo=float(lo[0]), rho_log_hi=float(hi[0]),
                rho_rank=est["rho_rank"], rho_rank_lo=float(lo[1]), rho_rank_hi=float(hi[1]),
                share_med=est["share"], share_lo=float(lo[2]), share_hi=float(hi[2]))


def conformal_q(scores, L):
    k = int(math.ceil((len(scores) + 1) * L)); s = np.sort(scores)
    return float(s[min(k, len(s)) - 1])


def run_cell(n, width, noise):
    sigma = noise / 100 * STD_CLEAN
    xt, it, yt = make_data(n, sigma)
    ev = list(range(min(n, N_EVAL)))
    xq = torch.tensor(np.tile(XQ, (len(ev), 1))).to(DEVICE); iq = torch.tensor(np.repeat(ev, len(XQ))).to(DEVICE)
    xc = torch.tensor(np.concatenate([GRID[CAL[k]] for k in ev])).to(DEVICE)
    ic = torch.tensor(np.concatenate([np.full(len(CAL[k]), k) for k in ev])).to(DEVICE)
    yc = np.concatenate([CLEAN[k][CAL[k]] + sigma * EPS[k][CAL[k]] for k in ev])
    m = fit(width, n, xt, it, yt)
    with torch.no_grad():
        mu, lv = m(xq, iq); muc, lvc = m(xc, ic)
        trr = float(((m(xt, it)[0] - yt) ** 2).mean(1).sqrt().mean())
    mus = mu.cpu().numpy().reshape(M, len(ev), -1); vs = torch.exp(lv).cpu().numpy().reshape(M, len(ev), -1)
    muc, vc = muc.cpu().numpy(), torch.exp(lvc).cpu().numpy()
    mu, A, E = mus.mean(0), vs.mean(0), mus.var(0, ddof=1)
    tot, err, out = A + E, np.abs(mu - FQ[ev]), ~HQ
    S = cKDTree(GRID[OBS[0]]).query(XQ, k=1)[0]
    per = lambda f: float(np.mean([f(i) for i in range(len(ev))]))
    # ---- calibration: scores on held-out calibration pixels, test on fresh noisy targets at random points
    sdc = np.sqrt(vc.mean(0) + muc.var(0, ddof=1))
    sc = np.abs(yc - muc.mean(0)) / sdc
    ytest = FQ[ev] + sigma * EPSQ[ev]; r = np.abs(ytest - mu); sd = np.sqrt(tot)
    cov_raw, cov_conf, cov_raw_h, cov_conf_h, qs = [], [], [], [], []
    for L in LEVELS:
        zl, q = float(norm.ppf((1 + L) / 2)), conformal_q(sc, L); qs.append(q)
        cov_raw.append(float((r <= zl * sd)[:, out].mean())); cov_conf.append(float((r <= q * sd)[:, out].mean()))
        cov_raw_h.append(float((r <= zl * sd)[:, HQ].mean())); cov_conf_h.append(float((r <= q * sd)[:, HQ].mean()))
    i90 = LEVELS.index(0.9)
    ae = ae_correlation(A[:, out], E[:, out], XQ[out])
    return dict(**ae, n=n, width=width, noise=noise, n_train=n_points(n), sigma=sigma,
                A=per(lambda i: np.median(A[i][out])), E=per(lambda i: np.median(E[i][out])), tot=per(lambda i: np.median(tot[i][out])),
                A_hole=per(lambda i: np.median(A[i][HQ])), E_hole=per(lambda i: np.median(E[i][HQ])),
                err=per(lambda i: np.median(err[i][out])), err_hole=per(lambda i: np.median(err[i][HQ])),
                rmse=per(lambda i: np.sqrt(np.mean(err[i][out] ** 2))),
                rho_std_err=per(lambda i: rho(np.sqrt(tot[i]), err[i], out)), rho_std_err_hole=per(lambda i: rho(np.sqrt(tot[i]), err[i], HQ)),
                share_E=per(lambda i: np.median(E[i][out] / tot[i][out])),
                E0=float(np.median(E[0][out])), A0=float(np.median(A[0][out])), err0=float(np.median(err[0][out])),
                err0_hole=float(np.median(err[0][HQ])), E0_hole=float(np.median(E[0][HQ])), rho0=rho(np.sqrt(tot[0]), err[0], out),
                train_rmse=trr, S=float(np.median(S)),
                z_std=float(((ytest - mu) / sd)[:, out].std()), cov_raw=cov_raw, cov_conf=cov_conf, cov_raw_hole=cov_raw_h, cov_conf_hole=cov_conf_h,
                q90=qs[i90], scale90=qs[i90] / float(norm.ppf(0.95)), z_std_cal=float(((yc - muc.mean(0)) / sdc).std()))

