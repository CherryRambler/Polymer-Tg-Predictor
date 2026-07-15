"""
validate_data.py

Purpose: Check that every SMILES string in our raw CSV can actually be parsed
into a valid molecule by RDKit. This is step 1 of preprocessing for ANY
cheminformatics ML project — invalid SMILES will silently crash or corrupt
everything downstream if you don't catch them first.

Run: python3 src/validate_data.py
"""

import pandas as pd
from rdkit import Chem

RAW_PATH = "data/raw/polymer_tg_polymetrix.csv"

def validate_smiles(smiles: str) -> bool:
    """
    Try to parse a SMILES string into an RDKit molecule object.
    Chem.MolFromSmiles returns None if parsing fails (invalid chemistry
    or malformed string) rather than raising an error — so we explicitly
    check for None.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    df["is_valid"] = df["repeat_unit_smiles"].apply(validate_smiles)

    invalid_rows = df[~df["is_valid"]]
    if len(invalid_rows) > 0:
        print(f"\n{len(invalid_rows)} INVALID SMILES found:")
        for _, row in invalid_rows.iterrows():
            print(f"  - {row['polymer_name']}: {row['repeat_unit_smiles']}")
    else:
        print("\nAll SMILES valid.")

    print(f"\nValid rows: {df['is_valid'].sum()} / {len(df)}")

if __name__ == "__main__":
    main()