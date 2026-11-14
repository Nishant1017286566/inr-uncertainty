"""Run the study. Results are written to results/raw.npz and summarised by analyse.py.

Usage:
    python3 run_study.py            # full run
    python3 run_study.py --quick    # 5 draws, fewer M values, for a smoke test
"""

from __future__ import annotations
import sys, time, json
import numpy as np

import config as C
import data as D
import rfblr as R


def member_noise_estimator(xy_obs, resid2, alpha, n_features, scale, rng):
    """Each ensemble member estimates its own noise variance map from squared residuals.

    This mirrors the baseline study, where every network carried a Gaussian NLL head
    predicting a per-pixel variance. Here the same job is done by regressing squared
    residuals on the same feature family, then clipping to stay positive.
    """
    phi = R.make_features(n_features, scale, rng)
    Po = phi(xy_obs)
    A = Po.T @ Po
    A[np.diag_indices_from(A)] += alpha
    w = np.linalg.solve(A, Po.T @ resid2)
    floor = 1e-6

    def s2_hat(xy):
        return np.maximum(phi(xy) @ w, floor)

    return s2_hat


def one_draw(seed: int, noise_factor: float, m_values):
    rng = np.random.default_rng(seed)
    d = D.make_draw(rng, noise_factor)
    xy, y, obs = d["xy"], d["y"], d["obs"]
    xy_obs, y_obs, s2_obs = xy[obs], y[obs], d["noise_sd"][obs] ** 2

    out = {}

    # ---- exact posterior, single reference feature draw ----------------------
    phi_ref = R.make_features(C.N_FEATURES, C.FOURIER_SCALE,
                              np.random.default_rng(seed + 10_000))
    Po = phi_ref(xy_obs)
    mean, cov = R.fit_posterior(Po, y_obs, s2_obs, C.PRIOR_PRECISION)
    mu_exact, epi_exact = R.predict(phi_ref(xy), mean, cov)
    alea_exact = d["noise_sd"] ** 2

    out["epi_exact"] = epi_exact
    out["alea_exact"] = alea_exact
    out["err_exact"] = np.abs(mu_exact - d["truth"])

    # ---- ensemble estimates at each M ---------------------------------------
    max_m = max(m_values)
    members, phi_fns = [], []
    for j in range(max_m):
        mrng = np.random.default_rng(seed + 50_000 + j)
        phi_j = R.make_features(C.N_FEATURES, C.FOURIER_SCALE, mrng)
        Pj = phi_j(xy_obs)
        mj, cj = R.fit_posterior(Pj, y_obs, s2_obs, C.PRIOR_PRECISION)
        resid2 = (y_obs - Pj @ mj) ** 2
        s2_fn = member_noise_estimator(xy_obs, resid2, C.PRIOR_PRECISION,
                                       C.N_FEATURES, C.FOURIER_SCALE, mrng)
        members.append((mj, cj, s2_fn))
        phi_fns.append(phi_j)

    for M in m_values:
        mu_b, epi_hat, alea_hat = R.ensemble_estimate(members[:M], phi_fns[:M], xy)
        out[f"epi_hat_M{M}"] = epi_hat
        out[f"alea_hat_M{M}"] = alea_hat
        out[f"mu_hat_M{M}"] = mu_b

    # ---- conformal calibration on held-out pixels ---------------------------
    held = ~obs & ~d["hole"]
    idx = np.flatnonzero(held)
    rng.shuffle(idx)
    ncal = int(len(idx) * C.CALIB_FRACTION)
    cal, test = idx[:ncal], idx[ncal:]

    sd_pred = np.sqrt(epi_exact + alea_exact)
    resid = np.abs(d["y"] - mu_exact)
    scores = resid[cal] / np.maximum(sd_pred[cal], 1e-9)
    k = int(np.ceil((len(cal) + 1) * C.NOMINAL_COVERAGE))
    k = min(k, len(cal))
    q = np.sort(scores)[k - 1]

    def coverage(ix, scale):
        return float(np.mean(resid[ix] <= scale * sd_pred[ix]))

    hole_idx = np.flatnonzero(d["hole"])
    out["cov_raw_global"] = coverage(test, 1.645)      # 90% Gaussian
    out["cov_conf_global"] = coverage(test, q)
    out["cov_raw_hole"] = coverage(hole_idx, 1.645)
    out["cov_conf_hole"] = coverage(hole_idx, q)
    out["q"] = q
    return out


CONDITIONS = {"base": 1.0, "double": C.NOISE_DOUBLING_FACTOR}


def main():
    """Run one or both conditions.

    Each condition is checkpointed to its own file as soon as it finishes, so the two
    can be run as separate processes and a loss of one does not cost the other:
        python3 run_study.py --condition base
        python3 run_study.py --condition double
        python3 run_study.py --merge
    """
    quick = "--quick" in sys.argv
    m_values = [2, 5, 9] if quick else C.M_VALUES
    n_draws = 5 if quick else C.N_DRAWS

    if "--merge" in sys.argv:
        store = {"m_values": m_values, "n_draws": n_draws}
        secs = 0.0
        for label in CONDITIONS:
            z = np.load(f"results/raw_{label}.npz")
            for k in z.files:
                if k == "seconds":
                    secs += float(z[k]); continue
                store[f"{label}__{k}"] = z[k]
        np.savez_compressed("results/raw.npz", **store)
        meta = {"quick": quick, "m_values": m_values, "n_draws": n_draws,
                "seconds": round(secs, 1)}
        with open("results/run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"merged -> results/raw.npz ({meta['seconds']}s of compute)")
        return

    if "--condition" in sys.argv:
        labels = [sys.argv[sys.argv.index("--condition") + 1]]
    else:
        labels = list(CONDITIONS)

    for label in labels:
        factor = CONDITIONS[label]
        t0 = time.time()
        per_draw = []
        for i in range(n_draws):
            per_draw.append(one_draw(C.SEED + i, factor, m_values))
            if (i + 1) % 5 == 0 or i == n_draws - 1:
                print(f"  {label}: {i+1}/{n_draws} draws  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        out = {k: np.stack([pd[k] for pd in per_draw]) for k in per_draw[0]}
        out["seconds"] = np.array(round(time.time() - t0, 1))
        np.savez_compressed(f"results/raw_{label}.npz", **out)
        print(f"  {label} saved -> results/raw_{label}.npz", flush=True)


if __name__ == "__main__":
    main()
