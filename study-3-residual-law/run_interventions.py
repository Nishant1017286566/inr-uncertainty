"""I1 (task difficulty) and I2 (optimisation budget): the causal tests of the residual law.

Uses the original study's own model, data and metrics, unmodified, imported from inr_core.
Only the two intervention knobs move. Baseline is the original study's baseline, so these
runs are directly comparable to its 97-setting grid.
"""
import json, os, sys, time
import numpy as np
import inr_core as C

os.makedirs("results", exist_ok=True)
N_SEED = 3
DIVERSITIES = [0.0, 0.5, 1.0, 2.0, 4.0]
STEPS_LIST = [150, 300, 600, 1000, 2000, 4000]
BASE = dict(C.BASE)            # n=10, width=50, noise=10


def one(seed, diversity=None, steps=None):
    """Run the original run_cell at the baseline, after setting a knob and rebuilding
    whatever depends on it. Returns the original study's own metric dict."""
    if diversity is not None and diversity != C.DIVERSITY:
        C.DIVERSITY = diversity
        # the image family depends on DIVERSITY: rebuild exactly as the original does
        C.CLEAN = np.stack([C.image_fn(C.GRID, k) for k in range(C.NMAX)])
        C.STD_CLEAN = float(C.CLEAN.std(axis=1).mean())
        C.FQ = np.stack([C.image_fn(C.XQ, k) for k in range(C.NMAX)])
    if steps is not None:
        C.STEPS = steps
    C.SEED_RUN = seed
    # reseed the ensemble init so seeds differ (the original fixes generator 1234)
    import torch
    torch.manual_seed(1234 + 1000 * seed)
    orig_init = C.EnsINR.__init__

    def init(self, width, n_img, M, _o=orig_init):
        g = torch.Generator().manual_seed(1234 + 1000 * seed)
        import math, torch.nn as nn
        nn.Module.__init__(self)
        self.register_buffer("B", torch.randn(M, 2, C.NFEAT, generator=g) * C.SCALE)
        self.emb = nn.Parameter(torch.randn(M, n_img, C.EMB, generator=g) * 0.3)
        dims = [2 * C.NFEAT + C.EMB] + [width] * C.LAYERS + [2]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for din, dout in zip(dims[:-1], dims[1:]):
            lim = 1 / math.sqrt(din)
            self.W.append(nn.Parameter((torch.rand(M, din, dout, generator=g) * 2 - 1) * lim))
            self.b.append(nn.Parameter((torch.rand(M, 1, dout, generator=g) * 2 - 1) * lim))
    C.EnsINR.__init__ = init
    try:
        r = C.run_cell(**BASE)
    finally:
        C.EnsINR.__init__ = orig_init
    return r


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    t0 = time.time()
    out = {"I1": [], "I2": []}
    path = f"results/interventions_{which}.json"

    if which in ("I1", "both"):
        for d in DIVERSITIES:
            for s in range(N_SEED):
                r = one(s, diversity=d)
                r.update(intervention="I1", knob=d, seed=s, steps=C.STEPS,
                         std_clean=C.STD_CLEAN)
                out["I1"].append(r)
                print(f"  I1 diversity={d} seed={s}: A={r['A']:.3e} "
                      f"sigma^2={r['sigma']**2:.3e} train_rmse={r['train_rmse']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(out, open(path, "w"), indent=1)
        C.DIVERSITY = 1.0           # restore before I2
        C.CLEAN = np.stack([C.image_fn(C.GRID, k) for k in range(C.NMAX)])
        C.STD_CLEAN = float(C.CLEAN.std(axis=1).mean())
        C.FQ = np.stack([C.image_fn(C.XQ, k) for k in range(C.NMAX)])

    if which in ("I2", "both"):
        for st in STEPS_LIST:
            for s in range(N_SEED):
                r = one(s, steps=st)
                r.update(intervention="I2", knob=st, seed=s, steps=st,
                         std_clean=C.STD_CLEAN)
                out["I2"].append(r)
                print(f"  I2 steps={st} seed={s}: A={r['A']:.3e} "
                      f"sigma^2={r['sigma']**2:.3e} train_rmse={r['train_rmse']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(out, open(path, "w"), indent=1)

    json.dump(out, open(path, "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
