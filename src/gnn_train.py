"""
gnn_train.py

Purpose: Train the GNN on our pre-built polymer graphs and evaluate it,
same metrics (MAE, R2) as Phase 1 so results are directly comparable.

Run: python3 src/gnn_train.py    (run from the project root folder)
"""

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import random

from gnn_model import PolymerGNN

GRAPH_PATH = "data/processed/graph/polymer_graphs.pt"
MODEL_OUT = "models/gnn_tg.pt"

def split_data(graphs, train_frac=0.8, val_frac=0.1, seed=42):
    random.seed(seed)
    shuffled = graphs.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test

def evaluate(model, loader, device):
    """Run the model on a dataset (no gradient updates) and return metrics."""
    model.eval()
    preds, actuals, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.extend(out.cpu().numpy())
            actuals.extend(batch.y.cpu().numpy())
            names.extend(batch.polymer_name)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    return mae, r2, preds, actuals, names

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graphs = torch.load(GRAPH_PATH, weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    train_graphs, val_graphs, test_graphs = split_data(graphs)
    print(f"Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    node_feature_dim = graphs[0].x.shape[1]
    edge_feature_dim = graphs[0].edge_attr.shape[1]
    model = PolymerGNN(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    n_epochs = 300
    best_val_mae = float("inf")

    print("Starting training...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_graphs)

        if epoch % 10 == 0 or epoch == n_epochs:
            val_mae, val_r2, _, _, _ = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:3d} | Train Loss = {train_loss:.4f} | "
                  f"Val MAE = {val_mae:6.2f}°C | Val R² = {val_r2:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), MODEL_OUT)

    print("Loading best model...")
    model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
    test_mae, test_r2, test_preds, test_actuals, test_names = evaluate(model, test_loader, device)

    print("Final Test Results")
    print("------------------")
    print(f"MAE : {test_mae:.2f}°C")
    print(f"R²  : {test_r2:.4f}")
    print(f"Best model saved to: {MODEL_OUT}")

    results_df = pd.DataFrame({
        "polymer_name": test_names,
        "actual_Tg": test_actuals,
        "gnn_predicted_Tg": test_preds,
    })
    results_df.to_csv("results/gnn_test_predictions.csv", index=False)
    print("Saved test predictions to results/gnn_test_predictions.csv")

if __name__ == "__main__":
    main()