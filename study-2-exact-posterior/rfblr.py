"""Random-feature Bayesian linear regression, with exact aleatoric and epistemic terms.

Why this model class
--------------------
The baseline study fitted an implicit neural representation whose first layer was a set of
*fixed, untrained* random Fourier features. The only trained part was what sat on top. If
we keep the same fixed features and make the top layer linear, the posterior is available
in closed form — so the aleatoric and epistemic components are known exactly rather than
estimated from an ensemble.

That is the whole point: it lets us score an ensemble estimate against the truth, instead
of only checking whether it responds to injected causes.

Model
-----
    phi(x) in R^D         random Fourier features, phi_j(x) = sqrt(2/D) cos(w_j . x + b_j)
    w ~ N(0, alpha^-1 I)  prior over output weights
    y | x  ~ N(phi(x)^T w, s2(x))     known heteroscedastic noise

Posterior (with precision-weighted design matrix):
    S = (alpha I + Phi^T diag(1/s2) Phi)^-1
    m = S Phi^T (y / s2)

Predictive at a test coordinate:
    mean       phi*^T m
    epistemic  phi*^T S phi*        <- exact, no ensemble needed
    aleatoric  s2(x*)               <- exact, known by construction
"""

from __future__ import annotations
import numpy as np


def make_features(n_features: int, scale: float, rng: np.random.Generator):
    """Draw a random Fourier feature map for 2-D coordinates."""
    omega = rng.normal(0.0, scale, size=(2, n_features))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    norm = np.sqrt(2.0 / n_features)

    def phi(xy: np.ndarray) -> np.ndarray:
        # xy: (N, 2) -> (N, D)
        return norm * np.cos(xy @ omega + bias)

    return phi


def fit_posterior(Phi: np.ndarray, y: np.ndarray, s2: np.ndarray, alpha: float):
    """Exact Gaussian posterior over output weights.

    Phi (N, D), y (N,), s2 (N,) known noise variances.
    Returns (mean, cov) of the weight posterior.
    """
    D = Phi.shape[1]
    w = 1.0 / s2
    A = (Phi * w[:, None]).T @ Phi
    A[np.diag_indices_from(A)] += alpha
    # Solve rather than invert where possible; we need the full covariance for the
    # epistemic term, so a Cholesky solve against the identity is the stable route.
    L = np.linalg.cholesky(A)
    cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))
    mean = cov @ (Phi.T @ (y * w))
    return mean, cov


def predict(phi_test: np.ndarray, mean: np.ndarray, cov: np.ndarray):
    """Posterior predictive mean and *exact* epistemic variance at test points."""
    mu = phi_test @ mean
    # diag(phi S phi^T) without forming the full matrix
    epistemic = np.einsum("nd,de,ne->n", phi_test, cov, phi_test)
    return mu, epistemic


def ensemble_estimate(members, phi_test_fns, xy_test):
    """Mimic what a practitioner does with an ensemble.

    members: list of (mean, cov, s2_hat_fn) per member, each with its own feature draw.
    Returns (mu_bar, epistemic_hat, aleatoric_hat) using the standard estimator:
        epistemic_hat = variance of member means
        aleatoric_hat = mean of member predicted noise variances
    """
    mus, a_hats = [], []
    for (mean, _cov, s2_hat_fn), phi_fn in zip(members, phi_test_fns):
        pt = phi_fn(xy_test)
        mus.append(pt @ mean)
        a_hats.append(s2_hat_fn(xy_test))
    mus = np.stack(mus)            # (M, N)
    a_hats = np.stack(a_hats)      # (M, N)
    mu_bar = mus.mean(axis=0)
    # ddof=1: unbiased, matching the usual ensemble-variance estimator
    epistemic_hat = mus.var(axis=0, ddof=1) if mus.shape[0] > 1 else np.zeros(mus.shape[1])
    aleatoric_hat = a_hats.mean(axis=0)
    return mu_bar, epistemic_hat, aleatoric_hat
