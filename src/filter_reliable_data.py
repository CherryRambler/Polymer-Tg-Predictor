"""
filter_reliable_data.py

Purpose: Remove genuinely unreliable polymer entries before training.

IMPORTANT CORRECTION from the first version of this script: "black"
reliability means "single source, can't be cross-checked" -- NOT "known
to be wrong." Dropping it removed 96% of our data for the wrong reason.
"red" is the category that actually means "conflicting Tg values reported
across multiple sources, Z-score > 2" -- genuine unreliability. That's
what we should filter on instead.

Run: python3 src/filter_reliable_data.py
"""

import pandas as pd

IN_PATH = "data/raw/polymer_tg_polymetrix.csv"
OUT_PATH = "data/raw/polymer_tg_polymetrix_reliable.csv"

def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} total polymers")

    print("\nReliability breakdown:")
    print(df["meta_reliability"].value_counts())

    reliable = df[df["meta_reliability"] != "red"].reset_index(drop=True)

    reliable.to_csv(OUT_PATH, index=False)
    print(f"\nKept {len(reliable)} polymers (dropped {len(df) - len(reliable)} 'red'/conflicting reliability)")
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()