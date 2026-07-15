"""
featurize.py

Purpose: Convert each polymer's repeat-unit SMILES into numeric features
a model can learn from. We compute two feature sets:

  1. RDKit descriptors  -> interpretable physicochemical properties
  2. Morgan fingerprints -> substructure pattern bits (ECFP-like)

Both are saved to data/processed/ so downstream training scripts never
need to touch RDKit directly.

Run: python3 src/featurize.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

RAW_PATH = "data/raw/polymer_tg_polymetrix_reliable.csv"
OUT_PATH = "data/processed/featurized.csv"

# A curated subset of RDKit descriptors relevant to polymer physical behavior.
DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumRings": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "MolLogP": Descriptors.MolLogP,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}

def compute_descriptors(mol):
    """Compute our chosen RDKit descriptors for one molecule."""
    return {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}

def compute_morgan_fingerprint(mol, n_bits=256, radius=2):
    """
    Morgan fingerprint: for each atom, look at its local neighborhood up to
    `radius` bonds away, hash that substructure into one of n_bits positions.
    n_bits=256 (up from 128 when we had only 50 samples) -- with 7,367
    samples we can afford a richer, less lossy fingerprint without the
    same overfitting risk.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp)

def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    descriptor_rows = []
    fingerprint_rows = []

    for smiles in df["repeat_unit_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        descriptor_rows.append(compute_descriptors(mol))
        fingerprint_rows.append(compute_morgan_fingerprint(mol))

    desc_df = pd.DataFrame(descriptor_rows)
    fp_df = pd.DataFrame(fingerprint_rows, columns=[f"fp_{i}" for i in range(256)])

    final_df = pd.concat(
        [df[["polymer_name", "Tg_celsius"]], desc_df, fp_df],
        axis=1
    )

    final_df.to_csv(OUT_PATH, index=False)
    print(f"Saved featurized data to {OUT_PATH}")
    print(f"Shape: {final_df.shape} (rows, columns)")
    print(f"\nDescriptor columns: {list(desc_df.columns)}")

if __name__ == "__main__":
    main()