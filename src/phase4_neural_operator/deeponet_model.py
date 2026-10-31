"""
deeponet_model.py

Purpose: The network architecture for Phase 4 -- a DeepONet (Deep
Operator Network).

The Phase 3 PINN learns ONE function: C(x, t) for a single, fixed D. Want
a different D? Retrain from scratch. A DeepONet instead learns an
OPERATOR: a mapping from "which D" to "the whole C(x, t; D) solution
function", trained once across many D's. After training, evaluating a
brand-new D is a single forward pass -- no retraining.

Architecture (standard DeepONet, kept as simple as the PINN it replaces):
  - Branch net: encodes the input function/parameter (here, just the
    scalar D) into a length-p latent vector.
  - Trunk net: encodes the query coordinates (x, t) into a length-p
    latent vector.
  - Output: dot product of the two latent vectors (+ a bias), exactly
    like the original DeepONet paper (Lu et al., 2021).

Input scaling (normalize_inputs=True, used by the deployed model):
  D spans two orders of magnitude (0.01 to 1.0). Fed in raw, the network
  barely "sees" the difference between D=0.01 and D=0.02, which is where
  the old model was weakest. So the branch net gets log10(D) + 1 (which
  maps 0.01..1.0 onto -1..1) and the trunk net gets x and t rescaled to
  -1..1. Callers always pass raw D, x, t; the scaling happens inside.

  A model saved with normalize_inputs=True carries a small marker tensor
  ("input_scaling") in its weights file, so load_deeponet() can tell the
  two formats apart and never silently loads one as the other.
"""

import torch
import torch.nn as nn

L = 1.0
T_MAX = 5.0


class DeepONet(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden_layers=3, latent_dim=64, normalize_inputs=False):
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

        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            # Marker saved in the state_dict (see module docstring).
            self.register_buffer("input_scaling", torch.ones(1))

    def forward(self, D, x, t):
        """
        D: (batch, 1) diffusion coefficients
        x: (batch, 1) positions
        t: (batch, 1) times
        Returns: (batch,) predicted concentration C(x, t; D)
        """
        if self.normalize_inputs:
            D = torch.log10(D) + 1.0
            x = 2.0 * x / L - 1.0
            t = 2.0 * t / T_MAX - 1.0
        branch_out = self.branch(D)               # (batch, latent_dim)
        trunk_in = torch.cat([x, t], dim=1)
        trunk_out = self.trunk(trunk_in)           # (batch, latent_dim)
        return torch.sum(branch_out * trunk_out, dim=1) + self.bias


def load_deeponet(path, hidden_dim=64, n_hidden_layers=3, latent_dim=64):
    """Load saved DeepONet weights, picking the matching input format from
    the file itself. Returns the model in eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = DeepONet(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
                     latent_dim=latent_dim, normalize_inputs="input_scaling" in state)
    model.load_state_dict(state)  # strict: any mismatch raises instead of loading garbage
    model.eval()
    return model
