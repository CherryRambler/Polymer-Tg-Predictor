# Polymer Tg Predictor

A from-scratch machine learning project predicting polymer glass transition
temperature (Tg) from molecular structure, and modeling physical transport
behavior (diffusion) with physics-informed learning. Built as a first ML
project, progressing from classical ML baselines through Graph Neural
Networks to Physics-Informed Neural Networks, with Neural Operators
planned next.

**Status: Phase 3 (PINN) complete.**

## Setup

```bash
pip install rdkit pandas scikit-learn xgboost matplotlib joblib torch torch_geometric
```

(`polymetrix` is NOT required to run this pipeline -- we only used it once,
early on, to understand the source data's column structure. See "Data"
below for how to get the actual dataset.)

## Data

We use the PolyMetriX curated Tg dataset (7,367 real polymers, PSMILES +
experimental Tg + source reliability metadata).

**Getting the data:** the automatic PolyMetriX downloader pulls from
Zenodo, which can time out on restricted/firewalled networks. Download the
file manually in your browser instead:
https://zenodo.org/records/14980914/files/LAMALAB_CURATED_Tg_structured.csv?download=1

Save it to `data/raw/LAMALAB_CURATED_Tg_structured.csv`, then run
`src/phase1_baseline/load_local_tg_data.py` to convert it into our
standard format.

**Note on data quality:** PolyMetriX flags each entry's reliability
(gold/yellow/black/red). "Black" (96% of the dataset) means "single
source, unverified" -- NOT "known wrong" -- so we keep it. "Red" (4 rows)
means "conflicting values across multiple sources" -- we drop only this
category via `src/phase1_baseline/filter_reliable_data.py`.

## Project structure

polymer-tg-predictor/
├── data/
│ ├── raw/ # original downloaded/converted data
│ └── processed/ # featurized data (fingerprints, graphs)
├── src/
│ ├── phase1_baseline/ # classical ML: data prep, featurization, RF/XGBoost
│ ├── phase2_gnn/ # graph neural network
│ ├── phase3_pinn/ # physics-informed neural network
│ └── analysis/ # cross-phase comparison and ensembling
├── models/ # saved trained models (gitignored, regenerable)
├── results/ # metrics, predictions, diagnostic plots
└── requirements.txt


## Pipeline (run in order, from the project root)

```bash
# 1. Get and clean the data
python src/phase1_baseline/load_local_tg_data.py   # convert downloaded CSV to our format
python src/phase1_baseline/filter_reliable_data.py # drop 'red' (conflicting) entries
python src/phase1_baseline/validate_data.py        # check all SMILES parse correctly

# 2. Phase 1: classical ML baseline
python src/phase1_baseline/featurize.py             # SMILES -> RDKit descriptors + Morgan fingerprints
python src/phase1_baseline/train_baseline.py        # train + cross-validate Random Forest and XGBoost
python src/phase1_baseline/plot_results.py          # diagnostic plots
python src/phase1_baseline/inspect_worst_predictions.py  # look up metadata for worst misses

# 3. Phase 2: Graph Neural Network
python src/phase2_gnn/gnn_data.py                   # SMILES -> PyTorch Geometric graph objects
python src/phase2_gnn/gnn_train.py                  # train + evaluate the GNN
python src/analysis/compare_phase1_vs_gnn.py        # compare GNN vs XGBoost on the same polymers
python src/analysis/ensemble.py                     # average both models' predictions

# 4. Phase 3: Physics-Informed Neural Network (diffusion)
python src/phase3_pinn/pinn_train.py                # train on the diffusion PDE (Adam + L-BFGS)
python src/phase3_pinn/pinn_evaluate.py             # validate against the exact analytical solution
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

### Phase 3: Physics-Informed Neural Network (diffusion PDE)

Unlike Phases 1-2 (structure -> single property), Phase 3 predicts full
time-and-space behavior: how a drug concentration profile evolves as it
diffuses out of a polymer slab, governed by Fick's second law
(dC/dt = D * d2C/dx2). The network (`src/phase3_pinn/pinn_model.py`, a
tanh-activated MLP taking (x, t) -> C) was trained with NO example
solution curves -- only a three-part physics loss (PDE residual via
automatic differentiation, initial condition, boundary condition).

Training uses two phases: Adam for broad convergence, then L-BFGS
fine-tuning for precision (`src/phase3_pinn/pinn_train.py`) -- standard
PINN practice. Validated against the exact analytical solution (truncated
Fourier series, `src/phase3_pinn/pinn_analytical.py`).

**Result:**

| Metric | Value |
|---|---|
| Mean absolute error (excluding IC/BC corner conflict points) | 0.0062 |
| Max absolute error (excluding corner points) | 0.062 |
| Max absolute error (all points, including the corner) | ~0.66-0.99 (run-dependent) |

The network correctly matches the true diffusion behavior at essentially
every point in the domain. The "max error (all points)" number is
misleading on its own -- see Known Limitations below.

**Current limitation:** the diffusion coefficient D is hardcoded (D=0.1),
not yet predicted from polymer structure. Connecting Phase 1/2's
structure-to-property models to this PINN is a natural next step but
requires a diffusion-coefficient dataset, which we don't yet have.

## Known limitations

- A handful of Phase 1/2 predictions remain badly wrong (150-300°C error),
  concentrated in structurally unusual polymers (rare ring systems,
  uncommon heteroatoms like Se) with few similar neighbors in the
  training data.
- A few specific cases (e.g. `polymer_5771`: both models predict
  ~340-490°C vs an actual recorded value of 25°C) are suspected data/
  label errors rather than model failures -- not independently verified.
- We initially misdiagnosed the "black" reliability flag as meaning
  "unreliable" and nearly discarded 96% of the dataset over it; corrected
  after checking the actual value counts. Documenting this here as a
  reminder to verify assumptions about metadata before filtering on it.
- The PINN has a mathematically unavoidable error at the single point
  (x=0 or x=L, t=0), where the initial condition (C=1) and boundary
  condition (C=0) directly contradict each other -- a known limitation of
  this problem formulation, not a training bug. An earlier attempt at
  L-BFGS fine-tuning also caused the model to diverge at large t by
  overfitting to one fixed set of collocation points; fixed by resampling
  fresh points across multiple smaller L-BFGS rounds instead of one large
  call on frozen points.

## Roadmap

- [x] Phase 0: tooling setup (RDKit, sklearn, xgboost)
- [x] Phase 1: classical ML baseline (XGBoost MAE 27.5°C / R2 0.87)
- [x] Phase 2: Graph Neural Network (ensemble MAE 26.7°C / R2 0.855)
- [x] Phase 3: Physics-Informed Neural Network (diffusion PDE, mean error
      0.0062 vs. exact solution) -- not yet connected to Phase 1/2's
      structure-to-property predictions
- [ ] Phase 4: Neural Operators (DeepONet / FNO)