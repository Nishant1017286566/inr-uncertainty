"""Fixed configuration. Set before the first run; not tuned against results.

Every number here was chosen a priori on the reasoning given in the comment, not by
looking at an outcome. If any value changes after the first run, the change and its
reason must be recorded in RESULTS.md as an amendment.
"""

# ---- signal -----------------------------------------------------------------
GRID = 64                 # 64x64 coordinate grid; small enough for 30 draws x 7 M values
SIGNAL_KIND = "mixture"   # smooth blobs + a sharp-edged disc + a stripe band

# ---- noise ------------------------------------------------------------------
# Heteroscedastic: std rises left -> right, so the aleatoric term has a known spatial
# pattern to recover (as in the baseline study's "channel noise" map).
NOISE_STD_LEFT = 0.02
NOISE_STD_RIGHT = 0.12
NOISE_DOUBLING_FACTOR = 2.0   # for the H2 entanglement intervention

# ---- sampling / training support --------------------------------------------
SAMPLE_FRACTION = 0.08    # fraction of pixels observed
HOLE = (0.55, 0.80, 0.20, 0.45)   # (r0, r1, c0, c1) as fractions: an unobserved block
THIN_REGION = (0.00, 0.25, 0.00, 1.00)  # top band sampled at reduced density
THIN_FACTOR = 0.25

# ---- model: random Fourier features -----------------------------------------
# The baseline study's INR used fixed, untrained random Fourier features in its first
# layer, so this is the same function class with a closed-form output layer.
N_FEATURES = 512          # D
FOURIER_SCALE = 4.0       # omega ~ N(0, scale^2 I); fixed a priori, not swept for fit
PRIOR_PRECISION = 1e-2    # alpha

# ---- ensembles ---------------------------------------------------------------
# Each member draws its own random feature set, mimicking independent inits.
M_VALUES = [2, 3, 5, 9, 15, 25, 50]
M_REFERENCE = 9           # the baseline study's attribution ensemble size

# ---- replication -------------------------------------------------------------
N_DRAWS = 30              # independent data draws per configuration
BOOTSTRAP = 10_000
SEED = 20260720           # fixed; draws are SEED + draw_index

# ---- calibration -------------------------------------------------------------
CALIB_FRACTION = 0.30     # of held-out pixels used for conformal calibration
NOMINAL_COVERAGE = 0.90

# ---- acceptance thresholds (from preregistration.md) -------------------------
H1_EXPONENT_RANGE = (-0.65, -0.35)
H2_MIN_RATIO = 1.10
H4_GLOBAL_RANGE = (0.88, 0.92)
H4_HOLE_MAX = 0.88
H4_MIN_DRAWS = 25

# secondary (exploratory) analysis
S1_DRAWS = 10          # draws for the data-resample control in secondary.py
