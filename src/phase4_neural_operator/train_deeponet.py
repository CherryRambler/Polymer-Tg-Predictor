"""
train_deeponet.py

Purpose: Train the Phase 4 DeepONet ONCE, offline, and save it to
models/deeponet_diffusion.pt. Unlike the Phase 3 PINN (trained live,
per-D, inside the Streamlit app), this script is meant to be run by hand
before deployment -- the app only ever loads the saved weights.

Run: python src/phase4_neural_operator/train_deeponet.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_model import DeepONet  # noqa: E402
from generate_training_data import generate_dataset  # noqa: E402

MODEL_OUT = "models/deeponet_diffusion.pt"


def train_deeponet(n_D=100, n_x=25, n_t=25, n_epochs=600, lr=2e-3,
                    batch_size=2048, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    D_vals, x_vals, t_vals, C_vals = generate_dataset(n_D=n_D, n_x=n_x, n_t=n_t, seed=seed)
    if verbose:
        print(f"Training set: {len(D_vals)} points across {n_D} D values.")

    D_t = torch.tensor(D_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    x_t = torch.tensor(x_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    t_t = torch.tensor(t_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    C_t = torch.tensor(C_vals, dtype=torch.float32, device=device)

    model = DeepONet(hidden_dim=64, n_hidden_layers=3, latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_points = D_t.shape[0]

    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_points, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_points, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            C_pred = model(D_t[idx], x_t[idx], t_t[idx])
            loss = loss_fn(C_pred, C_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch % 200 == 0 or epoch == n_epochs):
            print(f"Epoch {epoch:5d} | MSE = {epoch_loss / n_batches:.6f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    if verbose:
        print(f"\nSaved trained DeepONet to {MODEL_OUT}")

    return model


if __name__ == "__main__":
    train_deeponet()
