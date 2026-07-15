"""
plot_results.py

Purpose: Visualize cross-validated predictions vs actual values.
A single MAE number hides WHICH polymers are predicted well vs poorly --
this plot reveals patterns a metric can't (e.g., "the model consistently
underpredicts high-Tg rigid polymers").

Run: python3 src/plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/cv_predictions.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Plot 1: predicted vs actual ---
ax = axes[0]
ax.scatter(df["actual_Tg"], df["rf_predicted_Tg"], alpha=0.4, edgecolor="k", s=15)
lims = [df["actual_Tg"].min() - 20, df["actual_Tg"].max() + 20]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual Tg (°C)")
ax.set_ylabel("Predicted Tg (°C)")
ax.set_title("Random Forest: Predicted vs Actual Tg\n(5-fold cross-validated)")
ax.legend()
ax.grid(alpha=0.3)

# Label the worst-predicted points so we can inspect them by name
df["abs_error"] = (df["actual_Tg"] - df["rf_predicted_Tg"]).abs()
worst = df.nlargest(5, "abs_error")
for _, row in worst.iterrows():
    ax.annotate(str(row["polymer_name"]),
                (row["actual_Tg"], row["rf_predicted_Tg"]),
                fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")

# --- Plot 2: residuals ---
ax = axes[1]
residuals = df["actual_Tg"] - df["rf_predicted_Tg"]
ax.scatter(df["actual_Tg"], residuals, alpha=0.4, edgecolor="k", s=15)
ax.axhline(0, color="r", linestyle="--")
ax.set_xlabel("Actual Tg (°C)")
ax.set_ylabel("Residual (Actual - Predicted, °C)")
ax.set_title("Residuals vs Actual Tg")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/prediction_analysis.png", dpi=150)
print("Saved plot to results/prediction_analysis.png")

print("\nWorst 5 predictions:")
print(worst[["polymer_name", "actual_Tg", "rf_predicted_Tg", "abs_error"]].to_string(index=False))