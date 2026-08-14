"""
gnn_data.py

Purpose: Convert each polymer's SMILES string into a graph object PyTorch
Geometric can train on: atoms become nodes (with feature vectors), bonds
become edges. This replaces Phase 1's fingerprint featurization -- instead
of flattening structure into a checklist of yes/no substructure bits, we
keep the actual atom-bond connectivity intact for the model to learn from
directly.

We build ALL graphs once here and save them to disk, because building
graphs from 7,000+ molecules takes real time -- you never want to redo this
every time you tweak the model architecture or training loop.

Run: python3 src/gnn_data.py
"""
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem

RAW_PATH = "data/raw/polymer_tg_polymetrix_reliable.csv"   # your real dataset
OUT_PATH = "data/processed/graph/polymer_graphs.pt"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl", "Si", "Se", "*"]  # * = polymer connection point

def atom_features(atom):
    """
    Build a feature vector for one atom:
      - one-hot: which element is this?
      - degree: how many bonds does it have? (helps distinguish branching)
      - is_aromatic: 0/1
      - formal_charge: usually 0, but matters when nonzero
    """
    symbol = atom.GetSymbol()
    one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES]
    extra = [
        float(atom.GetDegree()),
        float(atom.GetIsAromatic()),
        float(atom.GetFormalCharge()),
    ]
    return one_hot + extra

def bond_features(bond):
    """Feature vector for one bond: one-hot bond type + is it in a ring?"""
    bond_type = bond.GetBondType()
    is_single = float(bond_type == Chem.BondType.SINGLE)
    is_double = float(bond_type == Chem.BondType.DOUBLE)
    is_triple = float(bond_type == Chem.BondType.TRIPLE)
    is_aromatic = float(bond_type == Chem.BondType.AROMATIC)
    is_in_ring = float(bond.IsInRing())
    return [is_single, is_double, is_triple, is_aromatic, is_in_ring]

def smiles_to_graph(smiles: str, tg_value: float):
    """
    Convert one SMILES string + its Tg label into a PyTorch Geometric
    Data object: x = node feature matrix, edge_index = connectivity,
    edge_attr = edge feature matrix, y = the target we're predicting.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    y = torch.tensor([tg_value], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    graphs = []
    failed = 0
    for _, row in df.iterrows():
        g = smiles_to_graph(row["repeat_unit_smiles"], row["Tg_celsius"])
        if g is None:
            failed += 1
            continue
        g.polymer_name = row["polymer_name"]
        graphs.append(g)

    print(f"Built {len(graphs)} graphs ({failed} failed to parse)")
    print(f"Example graph: {graphs[0]}")
    print(f"  Node feature dim: {graphs[0].x.shape[1]}")
    print(f"  Edge feature dim: {graphs[0].edge_attr.shape[1]}")

    torch.save(graphs, OUT_PATH)
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()