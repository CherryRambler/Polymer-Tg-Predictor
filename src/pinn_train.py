"""
pinn_train.py

Purpose: Train the PINN on the diffusion equation using a three-part loss:
1. PDE residual (physics) 2. Initial condition 3. Boundary conditions.
We generate fresh random points every epoch rather than looping over a
fixed dataset -- standard for PINNs.

Run: python3 src/pinn_train.py
"""

import torch
import numpy as np
import os

from pinn_model import PINN

D = 0.1
L = 1.0
T_MAX = 5.0
C0 = 1.0

MODEL_OUT = "models/pinn_diffusion.pt"
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

def pde_residual_loss(model, n_points, device):
    x = torch.rand(n_points, 1, device=device, requires_grad=True) * L
    t = torch.rand(n_points, 1, device=device, requires_grad=True) * T_MAX
    C = model(x, t)
    dC_dt = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    dC_dx = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0]
    residual = dC_dt - D * d2C_dx2
    return torch.mean(residual ** 2)

def initial_condition_loss(model, n_points, device):
    x = torch.rand(n_points, 1, device=device) * L
    t = torch.zeros(n_points, 1, device=device)
    C_pred = model(x, t)
    C_true = torch.full_like(C_pred, C0)
    return torch.mean((C_pred - C_true) ** 2)

def boundary_condition_loss(model, n_points, device):
    t = torch.rand(n_points, 1, device=device) * T_MAX
    x_left = torch.zeros(n_points, 1, device=device)
    x_right = torch.full((n_points, 1), L, device=device)
    C_left = model(x_left, t)
    C_right = model(x_right, t)
    return torch.mean(C_left ** 2) + torch.mean(C_right ** 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PINN(hidden_dim=64, n_hidden_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5000
    n_pde_points = 2000
    n_ic_points = 200
    n_bc_points = 200

    print("Starting training...")
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss_pde = pde_residual_loss(model, n_pde_points, device)
        loss_ic = initial_condition_loss(model, n_ic_points, device)
        loss_bc = boundary_condition_loss(model, n_bc_points, device)
        loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch:5d} | Total = {loss.item():.5f} | "
                  f"PDE = {loss_pde.item():.5f} | IC = {loss_ic.item():.5f} | "
                  f"BC = {loss_bc.item():.5f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nSaved trained PINN to {MODEL_OUT}")

if __name__ == "__main__":
    main()