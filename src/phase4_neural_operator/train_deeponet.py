"""
train_deeponet.py

Purpose: Train the Phase 4 DeepONet ONCE, offline, and save it. Unlike the
Phase 3 PINN (trained live, per-D, inside the Streamlit app), this script
is meant to be run by hand before deployment -- the app only ever loads
the saved weights.

Training recipes live in CONFIGS so every variant compared in
results/deeponet_before_after.md can be re-trained exactly:
  "original"  the first Phase 4 recipe (uniform D, raw inputs, 50-term
              series labels, 600 epochs)
  "log_d"     same budget, but log-uniform D sampling, log10(D) into the
              branch net and exact t = 0 labels
  "improved"  log_d + denser time points near t = 0 + more D values and
              epochs. This is the recipe of the deployed model.

Run (from the project root):
    python src/phase4_neural_operator/train_deeponet.py                 # deployed recipe
    python src/phase4_neural_operator/train_deeponet.py --config original --out models/x.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_model import DeepONet  # noqa: E402
from generate_training_data import generate_dataset  # noqa: E402

MODEL_OUT = "models/deeponet_diffusion.pt"
DEPLOYED_CONFIG = "improved"

CONFIGS = {
    "original": dict(n_D=100, n_x=25, n_t=25, n_epochs=600, d_sampling="uniform",
                     t_spacing="uniform", exact_labels=False, normalize_inputs=False),
    "log_d": dict(n_D=100, n_x=25, n_t=25, n_epochs=600, d_sampling="log_uniform",
                  t_spacing="uniform", exact_labels=True, normalize_inputs=True),
    "improved": dict(n_D=200, n_x=25, n_t=40, n_epochs=1500, d_sampling="log_uniform",
                     t_spacing="early", exact_labels=True, normalize_inputs=True),
}


def train_deeponet(n_D=100, n_x=25, n_t=25, n_epochs=600, lr=2e-3, batch_size=2048,
                   seed=0, d_sampling="uniform", t_spacing="uniform", exact_labels=False,
                   normalize_inputs=False, model_out=MODEL_OUT, verbose=True):
    """Train one DeepONet. Returns (model, timings) where timings holds the
    wall-clock seconds spent generating data and training."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    t0 = time.perf_counter()
    D_vals, x_vals, t_vals, C_vals = generate_dataset(
        n_D=n_D, n_x=n_x, n_t=n_t, seed=seed, d_sampling=d_sampling,
        t_spacing=t_spacing, exact_labels=exact_labels)
    data_seconds = time.perf_counter() - t0
    if verbose:
        print(f"Training set: {len(D_vals)} points across {n_D} D values "
              f"({data_seconds:.2f} s to generate).")

    D_t = torch.tensor(D_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    x_t = torch.tensor(x_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    t_t = torch.tensor(t_vals, dtype=torch.float32, device=device).reshape(-1, 1)
    C_t = torch.tensor(C_vals, dtype=torch.float32, device=device)

    model = DeepONet(hidden_dim=64, n_hidden_layers=3, latent_dim=64,
                     normalize_inputs=normalize_inputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_points = D_t.shape[0]

    t0 = time.perf_counter()
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
    train_seconds = time.perf_counter() - t0

    if model_out:
        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_out)
        if verbose:
            print(f"\nSaved trained DeepONet to {model_out} ({train_seconds:.1f} s training)")

    model.eval()
    return model, {"data_generation_seconds": data_seconds, "training_seconds": train_seconds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEPLOYED_CONFIG, choices=sorted(CONFIGS))
    parser.add_argument("--out", default=MODEL_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_deeponet(**CONFIGS[args.config], seed=args.seed, model_out=args.out)
