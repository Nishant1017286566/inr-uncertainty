"""Synthetic signal, heteroscedastic noise, and sampling with known support structure.

Everything a model could be uncertain about is injected here, so the ground truth for
both uncertainty components is known by construction:

  aleatoric  -> the noise std map, which rises left to right
  epistemic  -> driven by where samples are and are not: an unobserved block and a
                thinned band
"""

from __future__ import annotations
import numpy as np
import config as C


def coordinates(grid: int) -> np.ndarray:
    """(grid*grid, 2) coordinates on [-1, 1]^2, row-major."""
    lin = np.linspace(-1.0, 1.0, grid)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def signal(xy: np.ndarray) -> np.ndarray:
    """A mixture with smooth parts, a sharp edge, and oscillation.

    Deliberately not band-limited: the sharp disc edge is the part a smooth prior
    gets wrong, which is where misspecification shows up.
    """
    x, y = xy[:, 0], xy[:, 1]
    blobs = (np.exp(-((x + 0.45) ** 2 + (y + 0.35) ** 2) / 0.10)
             - 0.7 * np.exp(-((x - 0.5) ** 2 + (y - 0.45) ** 2) / 0.06))
    disc = 0.6 * ((x - 0.15) ** 2 + (y + 0.5) ** 2 < 0.07 ** 1.0).astype(float)
    stripes = 0.25 * np.sin(6.0 * np.pi * x) * (y > 0.55)
    return blobs + disc + stripes


def noise_std(xy: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Known heteroscedastic noise: std increases with x. Ground-truth aleatoric."""
    x = xy[:, 0]
    t = (x + 1.0) / 2.0                      # 0 at left, 1 at right
    base = C.NOISE_STD_LEFT + t * (C.NOISE_STD_RIGHT - C.NOISE_STD_LEFT)
    return base * factor


def _in_box(xy: np.ndarray, box) -> np.ndarray:
    """box = (r0, r1, c0, c1) as fractions of the [-1,1] square; rows map to y."""
    r0, r1, c0, c1 = box
    x, y = xy[:, 0], xy[:, 1]
    xf = (x + 1.0) / 2.0
    yf = (y + 1.0) / 2.0
    return (yf >= r0) & (yf < r1) & (xf >= c0) & (xf < c1)


def sample_mask(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Which pixels are observed. Encodes the unobserved block and the thinned band."""
    n = xy.shape[0]
    p = np.full(n, C.SAMPLE_FRACTION)
    p[_in_box(xy, C.THIN_REGION)] *= C.THIN_FACTOR
    p[_in_box(xy, C.HOLE)] = 0.0
    return rng.random(n) < p


def hole_mask(xy: np.ndarray) -> np.ndarray:
    return _in_box(xy, C.HOLE)


def make_draw(rng: np.random.Generator, noise_factor: float = 1.0):
    """One complete independent data draw."""
    xy = coordinates(C.GRID)
    f = signal(xy)
    sd = noise_std(xy, noise_factor)
    y = f + rng.normal(0.0, sd)
    obs = sample_mask(xy, rng)
    return {
        "xy": xy,
        "truth": f,
        "noise_sd": sd,
        "y": y,
        "obs": obs,
        "hole": hole_mask(xy),
    }
