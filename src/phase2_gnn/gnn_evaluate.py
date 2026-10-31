"""
gnn_evaluate.py

Purpose: Re-evaluate the COMMITTED GNN weights (models/gnn_tg.pt) on the
same 80/10/10 test split gnn_train.py uses (seed 42), and the 50/50
ensemble with XGBoost's cross-validated predictions on those polymers.
Saves the numbers to results/gnn_evaluation.json, so the README quotes the
model that is actually deployed rather than numbers from an older run.

The graphs are rebuilt from the committed CSV (same featurization and row
order as gnn_data.py), so this works in a fresh clone without the large,
uncommitted data/processed/graph/polymer_graphs.pt.

Run: python src/phase2_gnn/gnn_evaluate.py
"""

import json
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gnn_data import RAW_PATH, smiles_to_graph  # noqa: E402
from gnn_model import PolymerGNN  # noqa: E402
from gnn_train import evaluate, split_data  # noqa: E402

MODEL_PATH = "models/gnn_tg.pt"
CV_PATH = "results/cv_predictions.csv"
OUT_PATH = "results/gnn_evaluation.json"


def main():
    df = pd.read_csv(RAW_PATH)
    graphs = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row["repeat_unit_smiles"], row["Tg_celsius"])
        if g is not None:
            g.polymer_name = row["polymer_name"]
            graphs.append(g)
    _, _, test = split_data(graphs)

    model = PolymerGNN(node_feature_dim=12, edge_feature_dim=5, hidden_dim=64)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    gnn_mae, gnn_r2, preds, actuals, names = evaluate(model, DataLoader(test, batch_size=32), "cpu")

    res = pd.DataFrame({"polymer_name": names, "actual_Tg": [float(a) for a in actuals],
                        "gnn_predicted_Tg": [float(p) for p in preds]})
    res = res.merge(pd.read_csv(CV_PATH)[["polymer_name", "xgb_predicted_Tg"]], on="polymer_name")
    res["ensemble"] = (res["gnn_predicted_Tg"] + res["xgb_predicted_Tg"]) / 2

    def metrics(col):
        return {"mae": float(mean_absolute_error(res["actual_Tg"], res[col])),
                "r2": float(r2_score(res["actual_Tg"], res[col]))}

    out = {
        "model_file": MODEL_PATH,
        "split": "gnn_train.split_data, 80/10/10, random.seed(42)",
        "n_test": int(len(res)),
        "gnn": {"mae": float(gnn_mae), "r2": float(gnn_r2)},
        "xgboost_cv_on_same_polymers": metrics("xgb_predicted_Tg"),
        "ensemble_50_50": metrics("ensemble"),
        "note": "XGBoost predictions are 5-fold cross-validated (results/cv_predictions.csv). "
                "GNN training seed was not fixed in gnn_train.py, so retraining gives slightly "
                "different numbers.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
