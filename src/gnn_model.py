"""
gnn_model.py

Purpose: Define the actual Graph Neural Network architecture.

Architecture, in plain terms:
  1. Several "message passing" layers (GINEConv, not plain GCNConv): each
     round lets every atom update itself based on its bonded neighbors --
     AND the type of bond connecting them (single/double/aromatic, in a
     ring or not). Plain GCNConv ignores bond features entirely; GINEConv
     ("GIN" + Edge features) uses them, which matters a lot for Tg since
     ring rigidity is one of the biggest physical drivers of glass
     transition temperature.
  2. Global mean pooling: average all atoms' final vectors into ONE vector
     representing the whole molecule.
  3. A small ordinary neural network turns that vector into one predicted
     Tg number.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class PolymerGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64):
        super().__init__()

        self.edge_encoder = torch.nn.Linear(edge_feature_dim, hidden_dim)
        self.node_encoder = torch.nn.Linear(node_feature_dim, hidden_dim)

        def make_mlp():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        self.conv1 = GINEConv(make_mlp())
        self.conv2 = GINEConv(make_mlp())
        self.conv3 = GINEConv(make_mlp())

        self.fc1 = torch.nn.Linear(hidden_dim, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x.squeeze(-1)