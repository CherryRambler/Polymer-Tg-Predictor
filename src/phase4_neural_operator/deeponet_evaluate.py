"""
deeponet_evaluate.py

Purpose: Validate a trained DeepONet against the exact solution of the
problem as coded in this project, AT D VALUES NOT USED IN TRAINING.

The problem (same as Phase 3): Fick's second law dC/dt = D d2C/dx2 on a
slab x in [0, 1], t in [0, 5]; initial condition C(x, 0) = 1; boundary
conditions C(0, t) = C(1, t) = 0. Exact solution: the Fourier series in
pinn_analytical.py (generate_training_data.exact_solution adds the exact
value at t = 0, where the truncated series rings).

What is reported, separately for two sets of held-out D values:
  - interpolation: D inside the trained range [0.01, 1.0] (what the app's
    slider can produce). Training D values are random draws, so none of
    these fixed values is in the training set.
  - extrapolation: D outside the trained range (the app cannot produce
    these; reported to show where the operator stops being trustworthy).
Per D: MAE, relative L2 error, max error. Overall: MAE, relative L2, R2.

Corner rule (identical to pinn_evaluate.py): points with t <= 0.05 and
x < 0.03 or x > 0.97 are excluded from the error metrics, because there the
initial condition (C = 1) and boundary condition (C = 0) contradict each
other and no smooth function can match both.

Physical sanity: the true C always lies in [0, 1]. We report the fraction
of predictions below 0 or above 1 and the worst violation, both on the
evaluation grid (all points, corner included) and on exactly the curve the
app plots (x = 0.5, 200 times in [0, 5], every slider D from 0.01 to 1.00).

Run (from the project root):
    python src/phase4_neural_operator/deeponet_evaluate.py
    python src/phase4_neural_operator/deeponet_evaluate.py --model models/x.pt --out results/x.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_model import load_deeponet  # noqa: E402
from generate_training_data import D_MAX, D_MIN, exact_solution  # noqa: E402

MODEL_PATH = "models/deeponet_diffusion.pt"
RESULTS_PATH = "results/deeponet_evaluation.json"
L = 1.0
T_MAX = 5.0

# Fixed held-out D values. The first eight are the values the original
# Phase 4 evaluation used; the extra low values probe the low-D end, where
# the solution is hardest to learn.
INTERP_D_VALUES = [0.012, 0.02, 0.035, 0.06, 0.1, 0.15, 0.28, 0.42, 0.55, 0.68, 0.81, 0.95]
EXTRAP_D_VALUES = [0.003, 0.005, 0.0075, 1.25, 1.5, 2.0]

N_X, N_T = 51, 51
CORNER_T, CORNER_X = 0.05, 0.03


def eval_grid(n_x=N_X, n_t=N_T):
    xx, tt = np.meshgrid(np.linspace(0, L, n_x), np.linspace(0, T_MAX, n_t), indexing="ij")
    return xx.ravel(), tt.ravel()


def corner_mask(x, t):
    """True for points KEPT in the error metrics (same rule as the PINN)."""
    return ~((t <= CORNER_T) & ((x < CORNER_X) | (x > L - CORNER_X)))


def predict(model, D, x, t):
    with torch.no_grad():
        return model(
            torch.full((len(x), 1), float(D), dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1),
        ).numpy().astype(float)


def bounds_stats(pred):
    pred = np.asarray(pred, dtype=float)
    return {
        "frac_below_0": float(np.mean(pred < 0)),
        "frac_above_1": float(np.mean(pred > 1)),
        "min_pred": float(pred.min()),
        "max_pred": float(pred.max()),
        "worst_violation": float(max(0.0, -pred.min(), pred.max() - 1.0)),
    }


def evaluate_set(model, D_values):
    x, t = eval_grid()
    keep = corner_mask(x, t)
    per_D, preds, exacts, raw_preds = {}, [], [], []
    for D in D_values:
        p = predict(model, D, x, t)
        e = exact_solution(x, t, D)
        err = np.abs(p - e)[keep]
        per_D[f"{D:g}"] = {
            "mae": float(err.mean()),
            "rel_l2": float(np.linalg.norm((p - e)[keep]) / np.linalg.norm(e[keep])),
            "max_abs_error": float(err.max()),
        }
        preds.append(p[keep])
        exacts.append(e[keep])
        raw_preds.append(p)
    preds, exacts = np.concatenate(preds), np.concatenate(exacts)
    r2 = 1.0 - np.sum((exacts - preds) ** 2) / np.sum((exacts - exacts.mean()) ** 2)
    return {
        "D_values": list(D_values),
        "mae": float(np.mean(np.abs(preds - exacts))),
        "rel_l2": float(np.linalg.norm(preds - exacts) / np.linalg.norm(exacts)),
        "r2": float(r2),
        "worst_D_mae": float(max(v["mae"] for v in per_D.values())),
        "n_points": int(len(exacts)),
        "per_D": per_D,
        "bounds_on_grid": bounds_stats(np.concatenate(raw_preds)),
    }


def app_curve_bounds(model):
    """Bounds check on exactly what the app plots."""
    t = np.linspace(0, T_MAX, 200)
    x = np.full_like(t, 0.5)
    slider = np.round(np.arange(1, 101) * 0.01, 2)
    return bounds_stats(np.concatenate([predict(model, D, x, t) for D in slider]))


def evaluate(model):
    interp = evaluate_set(model, INTERP_D_VALUES)
    extrap = evaluate_set(model, EXTRAP_D_VALUES)
    return {
        # Top-level numbers are the interpolation set (the app's D range);
        # the app reads mae / r2 / D_values_tested.
        "mae": interp["mae"],
        "r2": interp["r2"],
        "rel_l2": interp["rel_l2"],
        "D_values_tested": interp["D_values"],
        "trained_D_range": [D_MIN, D_MAX],
        "grid": {"n_x": N_X, "n_t": N_T, "x_range": [0, L], "t_range": [0, T_MAX]},
        "corner_rule": f"excluded points with t <= {CORNER_T} and (x < {CORNER_X} or x > {L - CORNER_X})",
        "interpolation": interp,
        "extrapolation": extrap,
        "app_curve_bounds": app_curve_bounds(model),
    }


def print_summary(res, label=""):
    print(f"== {label}")
    for name in ("interpolation", "extrapolation"):
        s = res[name]
        print(f"  {name}: MAE {s['mae']:.4f}  rel L2 {s['rel_l2']:.4f}  R2 {s['r2']:.4f}  "
              f"worst-D MAE {s['worst_D_mae']:.4f}")
        print("    " + "  ".join(f"D={d}: {v['mae']:.4f}" for d, v in s["per_D"].items()))
    b = res["app_curve_bounds"]
    print(f"  app curve: {100 * b['frac_below_0']:.1f}% below 0, {100 * b['frac_above_1']:.1f}% above 1, "
          f"range [{b['min_pred']:.3f}, {b['max_pred']:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    results = evaluate(load_deeponet(args.model))
    results["model_file"] = args.model.replace("\\", "/")
    print_summary(results, args.model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")
