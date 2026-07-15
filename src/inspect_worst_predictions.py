"""
inspect_worst_predictions.py

Purpose: Look up the raw data + metadata (source, reliability, conflicting
Tg values) for our worst-predicted polymers, to check whether the model is
actually wrong or the LABEL is unreliable/conflicting.

Run: python3 src/inspect_worst_predictions.py
"""

import pandas as pd

raw = pd.read_csv("data/raw/polymer_tg_polymetrix.csv")
preds = pd.read_csv("results/cv_predictions.csv")

preds["abs_error"] = (preds["actual_Tg"] - preds["rf_predicted_Tg"]).abs()
worst = preds.nlargest(10, "abs_error")

merged = worst.merge(raw, on="polymer_name")

cols_to_show = ["polymer_name", "repeat_unit_smiles", "actual_Tg",
                 "rf_predicted_Tg", "abs_error"]
meta_cols = [c for c in merged.columns if c.startswith("meta_")]
cols_to_show += meta_cols

pd.set_option("display.max_colwidth", 60)
print(merged[cols_to_show].to_string(index=False))