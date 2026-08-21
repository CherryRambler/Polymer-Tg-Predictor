"""
pinn_evaluate.py

Purpose: Check the trained PINN against the known exact solution.

Run: python3 src/pinn_evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn_model import PINN
from pinn_analytical import analytical_solution

MODEL_PATH = "models/pinn_diffusion.pt"
D = 0.1
L = 1.0

def main():
    device = torch.device("cpu")
    model = PINN(hidden_dim=64, n_hidden_layers=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    x_plot = np.linspace(0, L, 200)
    time_snapshots = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_snapshots)))
    max_abs_error = 0.0
    all_errors = []
    all_errors_no_corner = []

    for t_val, color in zip(time_snapshots, colors):
        x_tensor = torch.tensor(x_plot, dtype=torch.float32).reshape(-1, 1)
        t_tensor = torch.full_like(x_tensor, t_val)
        with torch.no_grad():
            C_pinn = model(x_tensor, t_tensor).numpy().flatten()

        C_exact = analytical_solution(x_plot, np.full_like(x_plot, t_val), D=D, L=L)
        err = np.abs(C_pinn - C_exact)
        all_errors.extend(err)
        max_abs_error = max(max_abs_error, err.max())

        if t_val > 0.05:
            all_errors_no_corner.extend(err)
        else:
            near_corner = (x_plot < 0.03) | (x_plot > L - 0.03)
            all_errors_no_corner.extend(err[~near_corner])

        ax.plot(x_plot, C_exact, color=color, linewidth=2, label=f"t={t_val} (exact)")
        ax.plot(x_plot, C_pinn, color=color, linestyle="--", linewidth=1.5, label=f"t={t_val} (PINN)")

    ax.set_xlabel("Position x")
    ax.set_ylabel("Concentration C")
    ax.set_title("PINN vs analytical solution: diffusion out of a slab")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/pinn_vs_analytical.png", dpi=150)
    print("Saved plot to results/pinn_vs_analytical.png")
    all_errors = np.array(all_errors)
    all_errors_no_corner = np.array(all_errors_no_corner)

    print(f"\nMax absolute error (all points): {max_abs_error:.4f}")
    print(f"Mean absolute error (all points): {all_errors.mean():.4f}")
    print(f"Mean absolute error (excluding IC/BC corner conflict points): "
          f"{all_errors_no_corner.mean():.4f}")
    print(f"Max absolute error (excluding corner points): {all_errors_no_corner.max():.4f}")
    print("\n(The corner points at x=0 or x=1, t=0 have a mathematically"
          " unavoidable error since the initial condition C=1 and boundary"
          " condition C=0 contradict each other exactly there -- this is a"
          " known PINN limitation for this problem setup, not a training bug.)")

if __name__ == "__main__":
    main()