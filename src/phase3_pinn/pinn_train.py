"""
pinn_train.py

Purpose: Train the PINN on the diffusion equation using a three-part loss:
1. PDE residual (physics) 2. Initial condition 3. Boundary conditions.

Two-phase training: Adam first (fast, noisy, gets close), then L-BFGS
fine-tuning (slower, precise, polishes the fit) -- standard PINN practice.

LESSON LEARNED (kept here on purpose): an earlier version of the L-BFGS
phase used one FIXED set of points for the whole phase. It drove the loss
very low at exactly those points while letting the network diverge
elsewhere (e.g. blowing up at t=5 instead of staying near 0) -- classic
overfitting, just inside a physics loss instead of a normal dataset. Fixed
by resampling fresh points across multiple smaller L-BFGS rounds instead
of one large one.

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

    print(f"\nAdam phase done. Loss = {loss.item():.6f}")

    print("\nStarting L-BFGS fine-tuning...")
    n_lbfgs_rounds = 10
    for round_idx in range(1, n_lbfgs_rounds + 1):
        x_pde = torch.rand(n_pde_points, 1, device=device, requires_grad=True) * L
        t_pde = torch.rand(n_pde_points, 1, device=device, requires_grad=True) * T_MAX
        x_ic = torch.rand(n_ic_points, 1, device=device) * L
        t_ic = torch.zeros(n_ic_points, 1, device=device)
        t_bc = torch.rand(n_bc_points, 1, device=device) * T_MAX
        x_bc_left = torch.zeros(n_bc_points, 1, device=device)
        x_bc_right = torch.full((n_bc_points, 1), L, device=device)

        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=50,
            history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs_optimizer.zero_grad()
            C_pde = model(x_pde, t_pde)
            dC_dt = torch.autograd.grad(C_pde, t_pde, grad_outputs=torch.ones_like(C_pde), create_graph=True)[0]
            dC_dx = torch.autograd.grad(C_pde, x_pde, grad_outputs=torch.ones_like(C_pde), create_graph=True)[0]
            d2C_dx2 = torch.autograd.grad(dC_dx, x_pde, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0]
            l_pde = torch.mean((dC_dt - D * d2C_dx2) ** 2)
            C_ic = model(x_ic, t_ic)
            l_ic = torch.mean((C_ic - C0) ** 2)
            l_bc = torch.mean(model(x_bc_left, t_bc) ** 2) + torch.mean(model(x_bc_right, t_bc) ** 2)
            total = l_pde + 10.0 * l_ic + 10.0 * l_bc
            total.backward(retain_graph=True)
            return total

        lbfgs_optimizer.step(closure)
        with torch.enable_grad():
            round_loss = closure()
        print(f"L-BFGS round {round_idx:2d}/{n_lbfgs_rounds} | Loss = {round_loss.item():.6f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nSaved trained PINN to {MODEL_OUT}")

if __name__ == "__main__":
    main()