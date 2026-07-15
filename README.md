# Polymer Tg Predictor

A from-scratch machine learning project predicting polymer glass transition
temperature (Tg) from molecular structure. Built as a first ML project,
progressing from classical ML baselines to Graph Neural Networks, with
Physics-Informed AI and Neural Operators planned next.

**Status: Phase 2 (GNN) complete.**

## Setup

```bash
pip install rdkit pandas scikit-learn xgboost matplotlib joblib polymetrix torch torch_geometric
```

## Data

We use the PolyMetriX curated Tg dataset (7,367 real polymers, PSMILES +
experimental Tg + source reliability metadata).

**Getting the data:** the automatic PolyMetriX downloader pulls from
Zenodo, which can time out on restricted/firewalled networks. If
`src/get_data.py` fails, download the file manually in your browser: https://zenodo.org/records/14980914/files/LAMALAB_CURATED_Tg_structured.csv?download=1

Save it to `data/raw/LAMALAB_CURATED_Tg_structured.csv`, then run
`src/load_local_tg_data.py` instead of `get_data.py` to convert it into
our standard format.

**Note on data quality:** PolyMetriX flags each entry's reliability
(gold/yellow/black/red). "Black" (96% of the dataset) means "single
source, unverified" -- NOT "known wrong" -- so we keep it. "Red" (4 rows)
means "conflicting values across multiple sources" -- we drop only this
category via `src/filter_reliable_data.py`.

## Pipeline (run in order)

```bash
# 1. Get and clean the data
python src/load_local_tg_data.py      # convert downloaded CSV to our format
python src/filter_reliable_data.py    # drop 'red' (conflicting) entries
python src/validate_data.py           # check all SMILES parse correctly

# 2. Phase 1: classical ML baseline
python src/featurize.py               # SMILES -> RDKit descriptors + Morgan fingerprints
python src/train_baseline.py          # train + cross-validate Random Forest and XGBoost
python src/plot_results.py            # diagnostic plots
python src/inspect_worst_predictions.py  # look up metadata for worst misses

# 3. Phase 2: Graph Neural Network
python src/gnn_data.py                # SMILES -> PyTorch Geometric graph objects
python src/gnn_train.py               # train + evaluate the GNN
python src/compare_phase1_vs_gnn.py   # compare GNN vs XGBoost on the same polymers
python src/ensemble.py                # average both models' predictions
```

## Results

### Phase 1: Random Forest / XGBoost on Morgan fingerprints

5-fold cross-validated on 7,363 polymers:

| Model | MAE | R2 |
|---|---|---|
| Dummy (predict the mean) | 95.0°C | -- |
| Random Forest | 29.9°C | 0.854 |
| XGBoost | 27.5°C | 0.87 |

### Phase 2: Graph Neural Network (GINEConv)

Single 80/10/10 train/val/test split (7,363 polymers). Uses bond type and
ring-membership features via message passing, instead of a fixed
fingerprint checklist.

Evaluated on the same 737-polymer test set as XGBoost's corresponding
subset:

| Model | MAE | R2 |
|---|---|---|
| XGBoost (Phase 1) | 29.3°C | 0.836 |
| GNN (Phase 2) | 29.6°C | 0.825 |
| **Ensemble (average of both)** | **26.7°C** | **0.855** |

The GNN alone roughly ties XGBoost overall, but improved on 7 of
XGBoost's 10 worst individual predictions -- evidence it handles
structurally rare polymers (no close fingerprint match) better, even
without winning on average. Averaging both models' predictions beats
either one alone, confirming they make different, partially-cancelling
errors.

## Known limitations

- A handful of predictions remain badly wrong (150-300°C error) across
  both models, concentrated in structurally unusual polymers (rare ring
  systems, uncommon heteroatoms like Se) with few similar neighbors in
  the training data.
- A few specific cases (e.g. `polymer_5771`: both models predict
  ~340-490°C vs an actual recorded value of 25°C) are suspected data/
  label errors rather than model failures -- not independently verified.
- We initially misdiagnosed the "black" reliability flag as meaning
  "unreliable" and nearly discarded 96% of the dataset over it; corrected
  after checking the actual value counts. Documenting this here as a
  reminder to verify assumptions about metadata before filtering on it.

## Project structure

polymer-tg-predictor/
├── data/
│   ├── raw/          # original downloaded/converted data
│   └── processed/    # featurized data (fingerprints, graphs)
├── src/               # all pipeline code
├── models/            # saved trained models
├── results/           # metrics, predictions, diagnostic plots
└── requirements.txt

## Roadmap

- [x] Phase 0: tooling setup (RDKit, sklearn, xgboost)
- [x] Phase 1: classical ML baseline (XGBoost MAE 27.5°C / R2 0.87)
- [x] Phase 2: Graph Neural Network (ensemble MAE 26.7°C / R2 0.855)
- [ ] Phase 3: Physics-Informed AI (PINNs for diffusion/transport)
- [ ] Phase 4: Neural Operators (DeepONet / FNO)