"""
deeponet_model.py

Purpose: The network architecture for Phase 4 -- a DeepONet (Deep
Operator Network).

The Phase 3 PINN learns ONE function: C(x, t) for a single, fixed D. Want
a different D? Retrain from scratch (~20-30 seconds). A DeepONet instead
learns an OPERATOR: a mapping from "which D" to "the whole C(x, t; D)
solution function", trained once across many D's. After training,
evaluating a brand-new D is a single forward pass -- no retraining.

Architecture (standard DeepONet, kept as simple as the PINN it replaces):
  - Branch net: encodes the input function/parameter (here, just the
    scalar D) into a length-p latent vector.
  - Trunk net: encodes the query coordinates (x, t) into a length-p
    latent vector.
  - Output: dot product of the two latent vectors (+ a bias), exactly
    like the original DeepONet paper (Lu et al., 2021). This dot-product
    combination is what makes it an "operator" network rather than an
    ordinary MLP -- it factorizes "which input function" from "where do
    you want to query it".

Same hidden_dim/activation choices as pinn_model.py (tanh, since we're
still approximating a smooth PDE solution) so the two are easy to
compare architecturally.
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden_layers=3, latent_dim=64):
        super().__init__()

        def make_mlp(in_dim):
            layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            for _ in range(n_hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers += [nn.Linear(hidden_dim, latent_dim)]
            return nn.Sequential(*layers)

        # Branch net: input is just D (one scalar parameter per sample).
        self.branch = make_mlp(1)
        # Trunk net: input is the query coordinates (x, t).
        self.trunk = make_mlp(2)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, D, x, t):
        """
        D: (batch, 1) diffusion coefficients
        x: (batch, 1) positions
        t: (batch, 1) times
        Returns: (batch,) predicted concentration C(x, t; D)
        """
        branch_out = self.branch(D)               # (batch, latent_dim)
        trunk_in = torch.cat([x, t], dim=1)
        trunk_out = self.trunk(trunk_in)           # (batch, latent_dim)
        return torch.sum(branch_out * trunk_out, dim=1) + self.bias
