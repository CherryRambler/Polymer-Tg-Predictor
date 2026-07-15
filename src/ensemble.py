"""
ensemble.py

Purpose: Test whether averaging the XGBoost (Phase 1) and GNN (Phase 2)
predictions together beats either model alone. This works when the two
models make DIFFERENT kinds of mistakes -- averaging cancels out some of
each model's individual errors. It's one of the simplest, cheapest things
to try in ML before reaching for anything more complex.

Run: python3 src/ensemble.py
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

phase1 = pd.read_csv("results/cv_predictions.csv")
phase2 = pd.read_csv("results/gnn_test_predictions.csv")

merged = phase2.merge(
    phase1[["polymer_name", "xgb_predicted_Tg"]],
    on="polymer_name", how="inner"
)

merged["ensemble_predicted_Tg"] = (
    merged["xgb_predicted_Tg"] + merged["gnn_predicted_Tg"]
) / 2

xgb_mae = mean_absolute_error(merged["actual_Tg"], merged["xgb_predicted_Tg"])
gnn_mae = mean_absolute_error(merged["actual_Tg"], merged["gnn_predicted_Tg"])
ens_mae = mean_absolute_error(merged["actual_Tg"], merged["ensemble_predicted_Tg"])

xgb_r2 = r2_score(merged["actual_Tg"], merged["xgb_predicted_Tg"])
gnn_r2 = r2_score(merged["actual_Tg"], merged["gnn_predicted_Tg"])
ens_r2 = r2_score(merged["actual_Tg"], merged["ensemble_predicted_Tg"])

print(f"On {len(merged)} polymers (same test set for all three):\n")
print(f"{'Model':<12} {'MAE (°C)':>10} {'R2':>8}")
print(f"{'XGBoost':<12} {xgb_mae:>10.1f} {xgb_r2:>8.3f}")
print(f"{'GNN':<12} {gnn_mae:>10.1f} {gnn_r2:>8.3f}")
print(f"{'Ensemble':<12} {ens_mae:>10.1f} {ens_r2:>8.3f}")

merged.to_csv("results/ensemble_predictions.csv", index=False)
print("\nSaved to results/ensemble_predictions.csv")