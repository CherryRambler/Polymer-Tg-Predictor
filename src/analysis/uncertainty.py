"""
uncertainty.py

Purpose: Honest uncertainty for the app's Tg prediction.

1. Split-conformal 90% interval for the 50/50 ensemble.
   Built ONLY from out-of-sample residuals (never training residuals):
   results/ensemble_predictions.csv holds 737 polymers that the GNN never
   trained on (its test split) and whose XGBoost predictions come from
   5-fold cross-validation (each made by a model that never saw that
   polymer). Those 737 are split, with a fixed seed, into a calibration
   half and an evaluation half. The half-width is the
   ceil((n_cal + 1) * 0.9)-th smallest |error| on the calibration half (the
   finite-sample-corrected conformal quantile); the evaluation half then
   measures how often the interval really contains the true Tg.

2. Two warnings, each with a threshold taken from the calibration half and
   checked on the evaluation half:
   (a) the XGBoost and GNN predictions disagree a lot, and
   (b) the polymer is unlike anything in the training data (max Tanimoto
       similarity, same Morgan fingerprint as Phase 1: radius 2, 256 bits).

3. Context for novel chemistry: the interval above is calibrated on a
   RANDOM split, which is optimistic for structurally new polymers.
   Scaffold-split out-of-sample predictions exist only for XGBoost
   (results/scaffold_split_predictions.csv, from scaffold_split_eval.py),
   not for the GNN, so a scaffold-calibrated ensemble interval cannot be
   built. Instead we report how the random-split interval holds up on
   XGBoost's scaffold-split errors.

Run (from the project root; after scaffold_split_eval.py):
    python src/analysis/uncertainty.py        -> results/uncertainty.json
"""

import json
import math
import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ENSEMBLE_PATH = "results/ensemble_predictions.csv"
SCAFFOLD_PATH = "results/scaffold_split_predictions.csv"
RAW_PATH = "data/raw/polymer_tg_polymetrix_reliable.csv"
OUT_PATH = "results/uncertainty.json"

ALPHA = 0.10          # 90% intervals
SPLIT_SEED = 0        # calibration / evaluation split
GNN_SPLIT_SEED = 42   # gnn_train.split_data's seed (to rebuild its training split)
DISAGREE_PCTL = 90    # flag the top 10% of XGBoost-vs-GNN disagreement
SIMILARITY_PCTL = 10  # flag the 10% least similar to the training data
MORGAN_RADIUS, MORGAN_BITS = 2, 256   # same as src/phase1_baseline/featurize.py


# ---------------------------------------------------------------- conformal
def conformal_half_width(abs_residuals, alpha=ALPHA):
    """Split-conformal half-width: the ceil((n+1)(1-alpha))-th smallest
    absolute residual. Infinite if n is too small for that rank to exist."""
    r = np.sort(np.abs(np.asarray(abs_residuals, dtype=float)))
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(r[k - 1])


def coverage(y_true, y_pred, half_width):
    """Fraction of points whose true value lies inside pred +/- half_width."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= half_width))


def interval(pred, half_width):
    return pred - half_width, pred + half_width


def split_cal_eval(n, seed=SPLIT_SEED):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2:]


# ---------------------------------------------------------------- similarity
_GEN = None


def morgan_fp(smiles):
    """Morgan bit vector (Phase 1 settings), or None if RDKit can't parse."""
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _GEN.GetFingerprint(mol)


def max_tanimoto(fp, reference_fps):
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)))


def gnn_train_names(all_names, seed=GNN_SPLIT_SEED, train_frac=0.8):
    """Rebuild gnn_train.split_data's training split from the CSV row order
    (the graph file was built in the same order, one graph per row)."""
    names = list(all_names)
    random.seed(seed)
    random.shuffle(names)
    return set(names[: int(len(names) * train_frac)])


# ---------------------------------------------------------------- analysis
def group_stats(abs_err, inside, mask):
    return {"n": int(mask.sum()),
            "mae": float(abs_err[mask].mean()) if mask.any() else None,
            "coverage": float(inside[mask].mean()) if mask.any() else None}


def build():
    ens = pd.read_csv(ENSEMBLE_PATH)
    raw = pd.read_csv(RAW_PATH)
    ens = ens.merge(raw[["polymer_name", "repeat_unit_smiles"]], on="polymer_name", how="left")
    y = ens["actual_Tg"].to_numpy(float)
    pred = ens["ensemble_predicted_Tg"].to_numpy(float)
    xgb, gnn = ens["xgb_predicted_Tg"].to_numpy(float), ens["gnn_predicted_Tg"].to_numpy(float)
    cal, ev = split_cal_eval(len(ens))

    hw = conformal_half_width(np.abs(y[cal] - pred[cal]))
    hw_xgb = conformal_half_width(np.abs(y[cal] - xgb[cal]))
    abs_err = np.abs(y - pred)
    inside = abs_err <= hw

    # (a) disagreement
    disagree = np.abs(xgb - gnn)
    d_thr = float(np.percentile(disagree[cal], DISAGREE_PCTL))
    hi_d = disagree > d_thr

    # (b) similarity to the training data the held-out polymers were NOT in
    train = raw[raw["polymer_name"].isin(gnn_train_names(raw["polymer_name"]))]
    train_fps = [morgan_fp(s) for s in train["repeat_unit_smiles"]]
    sim = np.array([max_tanimoto(morgan_fp(s), train_fps) for s in ens["repeat_unit_smiles"]])
    s_thr = float(np.percentile(sim[cal], SIMILARITY_PCTL))
    lo_s = sim < s_thr

    ev_mask = np.zeros(len(ens), bool)
    ev_mask[ev] = True
    result = {
        "description": "Split-conformal 90% interval for the 50/50 XGBoost+GNN ensemble, from "
                       "out-of-sample residuals only (results/ensemble_predictions.csv).",
        "alpha": ALPHA, "nominal_coverage": 1 - ALPHA, "split_seed": SPLIT_SEED,
        "quantile_rule": "ceil((n_cal + 1) * 0.9)-th smallest |residual| on the calibration half",
        "n_cal": int(len(cal)), "n_eval": int(len(ev)),
        "half_width_C": hw,
        "empirical_coverage_eval": coverage(y[ev], pred[ev], hw),
        "xgboost_only": {"half_width_C": hw_xgb, "empirical_coverage_eval": coverage(y[ev], xgb[ev], hw_xgb)},
        "disagreement_warning": {
            "definition": "|XGBoost - GNN| prediction, in C",
            "threshold_C": d_thr, "threshold_rule": f"{DISAGREE_PCTL}th percentile on the calibration half",
            "eval_above_threshold": group_stats(abs_err, inside, ev_mask & hi_d),
            "eval_at_or_below_threshold": group_stats(abs_err, inside, ev_mask & ~hi_d),
        },
        "similarity_warning": {
            "definition": "max Tanimoto similarity (Morgan r=2, 256 bits) to the training polymers",
            "reference_set": "GNN training split (5,890 polymers) for the held-out analysis; the app "
                             "compares against all 7,363 polymers, which the deployed XGBoost trained on",
            "threshold": s_thr, "threshold_rule": f"{SIMILARITY_PCTL}th percentile on the calibration half",
            "eval_below_threshold": group_stats(abs_err, inside, ev_mask & lo_s),
            "eval_at_or_above_threshold": group_stats(abs_err, inside, ev_mask & ~lo_s),
        },
    }

    if os.path.exists(SCAFFOLD_PATH):
        sc = pd.read_csv(SCAFFOLD_PATH)
        r = np.abs(sc["actual_Tg"].to_numpy(float) - sc["xgb_scaffold_predicted_Tg"].to_numpy(float))
        sc_cal, sc_ev = split_cal_eval(len(sc))
        result["scaffold_context"] = {
            "note": "Scaffold-split out-of-sample predictions exist for XGBoost only (no GNN), so no "
                    "scaffold-calibrated ensemble interval is built. These numbers show how much wider "
                    "errors are on structurally novel polymers.",
            "source": SCAFFOLD_PATH, "n": int(len(sc)),
            "xgb_scaffold_half_width_C": conformal_half_width(r[sc_cal]),
            "xgb_scaffold_coverage_eval_of_its_own_half_width": float(np.mean(r[sc_ev] <= conformal_half_width(r[sc_cal]))),
            "random_split_ensemble_half_width_coverage_on_xgb_scaffold_errors": float(np.mean(r <= hw)),
        }
    return result


def main():
    res = build()
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
