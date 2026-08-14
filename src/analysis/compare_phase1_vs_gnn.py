"""
compare_phase1_vs_gnn.py

Purpose: Directly compare Phase 1 (XGBoost) and Phase 2 (GNN) on the SAME
polymers, to check whether the GNN actually fixed Phase 1's specific
weakness (bad predictions on structurally rare polymers) -- not just
"which overall MAE is smaller," but "did we solve the actual problem we
built this for."

Run: python3 src/compare_phase1_vs_gnn.py
"""

import pandas as pd

phase1 = pd.read_csv("results/cv_predictions.csv")   # has rf_predicted_Tg, xgb_predicted_Tg
phase2 = pd.read_csv("results/gnn_test_predictions.csv")

# Only compare on polymers that appear in BOTH result sets (the GNN's test
# set is a subset of everything Phase 1 saw during its cross-validation)
merged = phase2.merge(
    phase1[["polymer_name", "xgb_predicted_Tg"]],
    on="polymer_name", how="inner"
)

merged["xgb_error"] = (merged["actual_Tg"] - merged["xgb_predicted_Tg"]).abs()
merged["gnn_error"] = (merged["actual_Tg"] - merged["gnn_predicted_Tg"]).abs()

print(f"Comparing {len(merged)} polymers present in both result sets\n")
print(f"XGBoost MAE on this subset: {merged['xgb_error'].mean():.1f}°C")
print(f"GNN MAE on this subset:     {merged['gnn_error'].mean():.1f}°C")

# The real test: look at Phase 1's worst cases specifically -- did the GNN
# do better or worse on exactly those?
worst_for_xgb = merged.nlargest(10, "xgb_error")
print("\nPhase 1's 10 worst cases -- how did the GNN do on the SAME polymers?")
print(worst_for_xgb[["polymer_name", "actual_Tg", "xgb_predicted_Tg",
                      "gnn_predicted_Tg", "xgb_error", "gnn_error"]].to_string(index=False))

improved = (worst_for_xgb["gnn_error"] < worst_for_xgb["xgb_error"]).sum()
print(f"\nGNN improved on {improved}/10 of XGBoost's worst cases")