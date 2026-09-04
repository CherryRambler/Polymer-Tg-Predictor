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

Sampling range: D in [0.01, 1.0], matching the range used by the
Drug Release tab's D slider and by permeability_to_D.py's target range,
so the DeepONet is valid across the exact D values the app can produce.
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


def generate_dataset(n_D=200, n_x=40, n_t=40, seed=0):
    """Sample n_D diffusion coefficients and, for each, an n_x by n_t grid
    of (x, t) query points, labelled with the exact analytical C(x, t; D).

    Returns three flat arrays (D_vals, x_vals, t_vals) and a matching
    C_vals array, all the same length (n_D * n_x * n_t).
    """
    rng = np.random.default_rng(seed)

    D_samples = rng.uniform(D_MIN, D_MAX, size=n_D)
    x_grid = np.linspace(0, L, n_x)
    t_grid = np.linspace(0, T_MAX, n_t)

    D_out, x_out, t_out, C_out = [], [], [], []
    for D in D_samples:
        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        xx_flat = xx.flatten()
        tt_flat = tt.flatten()
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
