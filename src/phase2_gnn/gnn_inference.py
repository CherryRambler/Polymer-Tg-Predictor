"""
gnn_inference.py

Purpose: Turn a raw SMILES string into a PyTorch Geometric graph the
PolymerGNN model can consume, and run inference with it.
"""

import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem

ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl", "Si", "Se", "*"]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def atom_features(atom):
    """One-hot over ATOM_TYPES + [degree, is_aromatic, formal_charge] -> 12-dim."""
    symbol = atom.GetSymbol()
    one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES]
    extra = [
        float(atom.GetDegree()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetFormalCharge()),
    ]
    return one_hot + extra


def bond_features(bond):
    """One-hot [single, double, triple, aromatic] + [is_in_ring] -> 5-dim."""
    bond_type = bond.GetBondType()
    one_hot = [1.0 if bond_type == t else 0.0 for t in BOND_TYPES]
    extra = [1.0 if bond.IsInRing() else 0.0]
    return one_hot + extra


def smiles_to_graph(smiles):
    """Build a torch_geometric.data.Data graph from one SMILES string.

    Returns None if RDKit can't parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        # Add both directions since the graph is undirected.
        edge_indices.append([i, j])
        edge_attrs.append(feats)
        edge_indices.append([j, i])
        edge_attrs.append(feats)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def predict_gnn_tg(smiles, model):
    """Predict Tg for one SMILES string using a trained PolymerGNN.

    Returns a plain float, or None if the SMILES couldn't be parsed.
    """
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    batch = Batch.from_data_list([graph])
    model.eval()
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return float(pred.item())
