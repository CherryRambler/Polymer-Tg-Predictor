"""
scaffold_split_eval.py

Purpose: Re-evaluate the Phase 1 model using a SCAFFOLD SPLIT instead of a
random split -- a much more honest test of generalization.

The problem with a random split: two nearly-identical polymers can end up
one in train, one in test. The model doesn't have to genuinely generalize
to predict the test one well -- it's basically seen something almost
identical already. This inflates every score we've reported so far.

A scaffold split groups chemically related polymers by their Murcko
scaffold (the core ring/framework structure, ignoring side chains) and
forces each WHOLE GROUP into either train or test, never split across
both. Standard practice in molecular ML, specifically because random
splits are known to overestimate real-world performance on this data.

KNOWN LIMITATION (found while testing this on a small sample): Murcko
scaffolds only capture RING systems. Simple chain polymers with no rings
at all (polyethylene, PVC, PMMA, etc.) all collapse into one "empty
scaffold" bucket, which is less informative for distinguishing between
very different acyclic polymers. Worth checking how much this matters on
the real dataset -- print the scaffold count vs. total row count and see.

Run: python3 src/phase1_baseline/scaffold_split_eval.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

RAW_PATH = "data/raw/polymer_tg_polymetrix_reliable.csv"  # your real dataset

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def compute_fingerprint(smiles, n_bits=256, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(generator.GetFingerprint(mol))

def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    df["scaffold"] = df["repeat_unit_smiles"].apply(get_scaffold)
    n_unique_scaffolds = df["scaffold"].nunique()
    print(f"{n_unique_scaffolds} unique scaffolds across {len(df)} polymers")
    print(f"(If this is much less than {len(df)}, random splits were "
          f"likely leaking similar structures across train/test)\n")

    X = np.array([compute_fingerprint(s) for s in df["repeat_unit_smiles"]])
    y = df["Tg_celsius"].values
    groups = df["scaffold"].values

    n_splits = min(5, n_unique_scaffolds)
    gkf = GroupKFold(n_splits=n_splits)

    all_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        all_preds[test_idx] = model.predict(X[test_idx])

    mae = mean_absolute_error(y, all_preds)
    r2 = r2_score(y, all_preds)

    print(f"Scaffold-split results ({n_splits}-fold):")
    print(f"  MAE = {mae:.1f}°C")
    print(f"  R2  = {r2:.3f}")
    print("\nCompare this against your existing RANDOM-split cross-validated")
    print("results (train_baseline.py) -- a meaningfully worse score here")
    print("is expected and healthy: it reveals the model's real")
    print("generalization ability, not an inflated random-split number.")

if __name__ == "__main__":
    main()