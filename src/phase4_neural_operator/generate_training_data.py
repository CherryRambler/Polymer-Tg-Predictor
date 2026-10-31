"""
generate_training_data.py

Purpose: Generate (D, x, t) -> C training triples for the DeepONet.

CHOICE OF GROUND TRUTH -- analytical solution, not train_pinn():
We have two possible sources of ground truth for "the diffusion PDE
solution at a given D": (a) run pinn_train.train_pinn() to train a fresh
PINN for each sampled D, or (b) evaluate pinn_analytical.analytical_solution
directly. We use (b), for two reasons:
  1. Correctness: the analytical solution is the EXACT solution (a known
     closed-form Fourier series for this slab-diffusion problem, per
     Crank's "The Mathematics of Diffusion"). A freshly-trained PINN is
     itself only an approximation of that same exact solution, with its
     own training noise/error on top. Training the DeepONet against exact
     values avoids inheriting and compounding a second layer of
     approximation error.
  2. Speed: analytical_solution() is a closed-form NumPy sum -- computing
     thousands of (D, x, t) points takes milliseconds. Training one PINN
     per D takes ~20-30 seconds; for even 50 D values that's ~20 minutes
     just to build the dataset, before the DeepONet itself is trained.

Sampling range: D in [0.01, 1.0], matching the range of the app's D
slider, so the DeepONet covers every D value the app can produce.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase3_pinn"))
# pyrefly: ignore [missing-import]
from pinn_analytical import analytical_solution  # noqa: E402

L = 1.0
T_MAX = 5.0
D_MIN, D_MAX = 0.01, 1.0


def exact_solution(x, t, D, L=L, n_terms=400):
    """The exact C(x, t; D), including at t = 0.

    The Fourier series converges slowly at t = 0 (a truncated series
    "rings" around the true value of 1 -- the Gibbs effect), so there we
    use the initial condition itself: C = 1 inside the slab, C = 0 on the
    two edges (the boundary condition wins at the corner, as it does in the
    series). For t > 0 the series is used with n_terms terms. With 400
    terms the first dropped term is smaller than 1e-20 whenever
    D * t >= 1e-5, which covers every (D, t) point this project uses.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    C = analytical_solution(x, np.where(t > 0, t, 1.0), D=D, L=L, n_terms=n_terms)
    C = np.where(t > 0, C, 1.0)
    C = np.where((x <= 0) | (x >= L), 0.0, C)
    return C


def generate_dataset(n_D=200, n_x=40, n_t=40, seed=0, d_sampling="uniform",
                     t_spacing="uniform", exact_labels=False):
    """Sample n_D diffusion coefficients and, for each, an n_x by n_t grid
    of (x, t) query points, labelled with the analytical C(x, t; D).

    Options (defaults reproduce the original Phase 4 recipe):
      d_sampling   "uniform" draws D uniformly from [D_MIN, D_MAX];
                   "log_uniform" draws log10(D) uniformly, so D = 0.01..0.1
                   gets as many samples as D = 0.1..1.0.
      t_spacing    "uniform" spaces t evenly; "early" uses t = T_MAX * s**2
                   for evenly spaced s, packing more points near t = 0 where
                   the solution changes fastest (a data-side way of weighting
                   the loss toward small t).
      exact_labels False uses the 50-term series everywhere (it rings at
                   t = 0); True uses exact_solution().

    Returns three flat arrays (D_vals, x_vals, t_vals) and a matching
    C_vals array, all the same length (n_D * n_x * n_t).
    """
    rng = np.random.default_rng(seed)

    if d_sampling == "uniform":
        D_samples = rng.uniform(D_MIN, D_MAX, size=n_D)
    elif d_sampling == "log_uniform":
        D_samples = 10 ** rng.uniform(np.log10(D_MIN), np.log10(D_MAX), size=n_D)
    else:
        raise ValueError(f"unknown d_sampling {d_sampling!r}")

    x_grid = np.linspace(0, L, n_x)
    if t_spacing == "uniform":
        t_grid = np.linspace(0, T_MAX, n_t)
    elif t_spacing == "early":
        t_grid = T_MAX * np.linspace(0, 1, n_t) ** 2
    else:
        raise ValueError(f"unknown t_spacing {t_spacing!r}")

    D_out, x_out, t_out, C_out = [], [], [], []
    for D in D_samples:
        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        xx_flat = xx.flatten()
        tt_flat = tt.flatten()
        if exact_labels:
            C_flat = exact_solution(xx_flat, tt_flat, D=D, L=L)
        else:
            C_flat = analytical_solution(xx_flat, tt_flat, D=D, L=L)

        D_out.append(np.full_like(xx_flat, D))
        x_out.append(xx_flat)
        t_out.append(tt_flat)
        C_out.append(C_flat)

    return (
        np.concatenate(D_out).astype(np.float32),
        np.concatenate(x_out).astype(np.float32),
        np.concatenate(t_out).astype(np.float32),
        np.concatenate(C_out).astype(np.float32),
    )


if __name__ == "__main__":
    D_vals, x_vals, t_vals, C_vals = generate_dataset()
    print(f"Generated {len(D_vals)} training points across "
          f"{len(np.unique(D_vals))} distinct D values.")
    print(f"D range: [{D_vals.min():.3f}, {D_vals.max():.3f}]")
    print(f"C range: [{C_vals.min():.3f}, {C_vals.max():.3f}]")
