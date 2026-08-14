"""
pinn_analytical.py

Purpose: The known, exact solution to our test problem -- a slab of
thickness L=1, initially at uniform concentration C0=1, both edges held
at C=0. This is a classic solved diffusion problem (Crank's "The
Mathematics of Diffusion"). We need it to verify the PINN actually
learned real physics, not just something that looks plausible.
"""

import numpy as np

def analytical_solution(x, t, D=0.1, L=1.0, C0=1.0, n_terms=50):
    C = np.zeros_like(x, dtype=float)
    for n in range(n_terms):
        k = 2 * n + 1
        coeff = 4 * C0 / (k * np.pi)
        C += coeff * np.sin(k * np.pi * x / L) * np.exp(-D * (k * np.pi / L) ** 2 * t)
    return C