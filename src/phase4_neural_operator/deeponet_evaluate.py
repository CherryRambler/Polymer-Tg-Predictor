"""
deeponet_evaluate.py

Purpose: Validate the trained DeepONet against pinn_analytical.py's exact
solution, AT D VALUES NOT SEEN DURING TRAINING. This is the real claim
Phase 4 makes -- not "it memorized the training D's" but "it generalizes
to interpolated D's it never trained on". train_deeponet.py samples D
uniformly at random from [0.01, 1.0]; this script picks a fixed,
different set of D's (on a regular grid, offset from the random training
samples) and reports the real MAE/R2 against the exact solution.

Run: python src/phase4_neural_operator/deeponet_evaluate.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_model import DeepONet  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase3_pinn"))
# pyrefly: ignore [missing-import]
from pinn_analytical import analytical_solution  # noqa: E402

MODEL_PATH = "models/deeponet_diffusion.pt"
RESULTS_PATH = "results/deeponet_evaluation.json"
L = 1.0
T_MAX = 5.0

# A fixed grid of held-out D values, deliberately NOT the random samples
# train_deeponet.py draws (those depend on the training seed) -- these
# specific values were never part of any training run.
TEST_D_VALUES = [0.02, 0.15, 0.28, 0.42, 0.55, 0.68, 0.81, 0.95]


def load_model():
    model = DeepONet(hidden_dim=64, n_hidden_layers=3, latent_dim=64)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model, D_values=TEST_D_VALUES, n_x=50, n_t=50):
    x_grid = np.linspace(0, L, n_x)
    t_grid = np.linspace(0, T_MAX, n_t)
    xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
    xx_flat = xx.flatten()
    tt_flat = tt.flatten()

    all_pred, all_exact = [], []
    per_D_mae = {}

    x_t = torch.tensor(xx_flat, dtype=torch.float32).reshape(-1, 1)
    t_t = torch.tensor(tt_flat, dtype=torch.float32).reshape(-1, 1)

    for D in D_values:
        C_exact = analytical_solution(xx_flat, tt_flat, D=D, L=L)
        D_t = torch.full((len(xx_flat), 1), D, dtype=torch.float32)
        with torch.no_grad():
            C_pred = model(D_t, x_t, t_t).numpy()

        mae = float(np.mean(np.abs(C_pred - C_exact)))
        per_D_mae[D] = mae
        all_pred.append(C_pred)
        all_exact.append(C_exact)

    all_pred = np.concatenate(all_pred)
    all_exact = np.concatenate(all_exact)

    mae = float(np.mean(np.abs(all_pred - all_exact)))
    ss_res = float(np.sum((all_exact - all_pred) ** 2))
    ss_tot = float(np.sum((all_exact - all_exact.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    return {
        "mae": mae,
        "r2": r2,
        "per_D_mae": per_D_mae,
        "n_points": len(all_exact),
        "D_values": list(D_values),
    }


if __name__ == "__main__":
    model = load_model()
    results = evaluate(model)

    print(f"Evaluated on {results['n_points']} points across "
          f"{len(results['D_values'])} held-out D values (not used in training):")
    print(f"  D values tested: {results['D_values']}")
    print()
    for D, mae in results["per_D_mae"].items():
        print(f"  D = {D:.2f}  ->  MAE = {mae:.4f}")
    print()
    print(f"Overall MAE: {results['mae']:.4f}")
    print(f"Overall R2:  {results['r2']:.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "mae": results["mae"],
            "r2": results["r2"],
            "n_points": results["n_points"],
            "D_values_tested": results["D_values"],
            "per_D_mae": {f"{D:.2f}": mae for D, mae in results["per_D_mae"].items()},
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")
