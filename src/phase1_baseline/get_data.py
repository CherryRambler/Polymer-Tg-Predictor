"""
load_local_tg_data.py

Purpose: Load the manually-downloaded PolyMetriX Tg dataset from disk,
bypassing the package's built-in downloader (which times out on
restricted/firewalled networks). Same output format as get_data.py.

Before running: download this file manually in your browser and save it to
data/raw/LAMALAB_CURATED_Tg_structured.csv
  https://zenodo.org/records/14980914/files/LAMALAB_CURATED_Tg_structured.csv?download=1

Run: python3 src/load_local_tg_data.py
"""

import pandas as pd

IN_PATH = "data/raw/LAMALAB_CURATED_Tg_structured.csv"
OUT_PATH = "data/raw/polymer_tg_polymetrix.csv"

def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} polymers from local file")

    out = pd.DataFrame({
        "polymer_name": [f"polymer_{i}" for i in range(len(df))],
        "repeat_unit_smiles": df["PSMILES"],
        "Tg_celsius": df["labels.Exp_Tg(K)"] - 273.15,
    })

    meta_cols = [c for c in df.columns if c.startswith("meta.")]
    for c in meta_cols:
        out[c.replace("meta.", "meta_")] = df[c]

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} polymers to {OUT_PATH}")
    print(f"Tg range: {out['Tg_celsius'].min():.1f}°C to {out['Tg_celsius'].max():.1f}°C")

if __name__ == "__main__":
    main()