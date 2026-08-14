"""
pinn_model.py

Purpose: The network architecture for the Physics-Informed Neural Network.

Unlike Phase 1/2, this is a genuinely simple ordinary MLP -- all the
"physics" happens in HOW we train it (pinn_train.py), not in the
architecture. Input: (x, t). Output: predicted concentration C.

We use tanh activation (not ReLU) because PINNs need SMOOTH,
infinitely-differentiable outputs -- we take second derivatives of this
network's output with respect to its inputs, and ReLU's sharp corner
breaks that.
"""

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)