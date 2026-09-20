"""
test_gnn.py

Purpose: Confirm the GNN's graph-building and model forward pass produce
correctly-shaped output, including on edge cases (single-atom molecules).

Note: smiles_to_graph() in gnn_data.py takes (smiles, tg_value) since
tg_value is the real training label. These tests pass a dummy tg_value
of 0.0 since the label itself isn't what's under test here.

Run: pytest tests/test_gnn.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "phase2_gnn"))

import torch
from torch_geometric.data import Batch
from gnn_data import smiles_to_graph
from gnn_model import PolymerGNN


def test_graph_built_for_valid_smiles():
    graph = smiles_to_graph("CC(c1ccccc1)C", 0.0)
    assert graph is not None
    assert graph.x.shape[1] == 12
    assert graph.edge_attr.shape[1] == 5


def test_invalid_smiles_returns_none():
    graph = smiles_to_graph("not a molecule", 0.0)
    assert graph is None


def test_single_atom_molecule_does_not_crash():
    graph = smiles_to_graph("C", 0.0)
    assert graph is not None
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 5)


def test_model_forward_pass_shape():
    graph1 = smiles_to_graph("CC(c1ccccc1)C", 0.0)
    graph2 = smiles_to_graph("CC", 0.0)
    batch = Batch.from_data_list([graph1, graph2])

    model = PolymerGNN(node_feature_dim=12, edge_feature_dim=5, hidden_dim=32)
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()
