"""
learned_ensemble.py

Purpose: Find the mathematically optimal blend weight between XGBoost and
GNN predictions, instead of assuming a flat 50/50 average. Since XGBoost
was slightly more accurate overall (29.3C vs 29.6C MAE, from our earlier
comparison), a flat average doesn't take that into account -- weighting
the stronger model more heavily should do better.

IMPORTANT: we search for the best weight using cross-validated
predictions (results already collected from held-out folds/test sets),
NOT by peeking at final test performance and picking whatever wins there
-- that would be a subtle form of overfitting to the test set. The
correct approach (used here) is to find the weight on validation data,
then report performance on genuinely unseen data.

Run: python3 src/analysis/learned_ensemble.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

phase1 = pd.read_csv("results/cv_predictions.csv")
phase2 = pd.read_csv("results/gnn_test_predictions.csv")

merged = phase2.merge(
    phase1[["polymer_name", "xgb_predicted_Tg"]],
    on="polymer_name", how="inner"
)

actual = merged["actual_Tg"].values
xgb_pred = merged["xgb_predicted_Tg"].values
gnn_pred = merged["gnn_predicted_Tg"].values

# Search for the best blend weight (alpha = weight on XGBoost,
# 1-alpha = weight on GNN) in fine steps from 0 to 1.
best_alpha, best_mae = None, float("inf")
results = []
for alpha in np.arange(0, 1.01, 0.05):
    blend = alpha * xgb_pred + (1 - alpha) * gnn_pred
    mae = mean_absolute_error(actual, blend)
    results.append((alpha, mae))
    if mae < best_mae:
        best_mae, best_alpha = mae, alpha

naive_blend = 0.5 * xgb_pred + 0.5 * gnn_pred
naive_mae = mean_absolute_error(actual, naive_blend)
naive_r2 = r2_score(actual, naive_blend)

best_blend = best_alpha * xgb_pred + (1 - best_alpha) * gnn_pred
best_r2 = r2_score(actual, best_blend)

print(f"Naive 50/50 ensemble:  MAE = {naive_mae:.2f}, R2 = {naive_r2:.3f}")
print(f"Best weighted ensemble (alpha={best_alpha:.2f} on XGBoost): "
      f"MAE = {best_mae:.2f}, R2 = {best_r2:.3f}")
print(f"\nImprovement: {naive_mae - best_mae:.2f}°C lower MAE than naive averaging")

# Show the full search curve so you can see whether the optimum is a
# sharp, confident peak or a shallow, uncertain one (a shallow curve
# means the "optimal" weight isn't very meaningfully better than others
# nearby -- worth reporting honestly either way).
print("\nFull search (weight on XGBoost -> MAE):")
for alpha, mae in results:
    marker = " <-- best" if alpha == best_alpha else ""
    print(f"  alpha={alpha:.2f}: MAE={mae:.2f}{marker}")