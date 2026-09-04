"""
app.py

Purpose: A Streamlit web app frontend for the whole project -- lets a
person type in a polymer's SMILES string and get:
  1. Predicted glass transition temperature (Tg), from the Phase 1 model
  2. A predicted drug-release curve, from the Phase 3 PINN, using a
     structure-derived diffusion coefficient

Run locally:
    streamlit run app.py
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "phase3_pinn"))
# pyrefly: ignore [missing-import]
from pinn_train import train_pinn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "phase2_gnn"))

GNN_AVAILABLE = True
try:
    # pyrefly: ignore [missing-import]
    from gnn_model import PolymerGNN  # noqa: E402
    # pyrefly: ignore [missing-import]
    from gnn_inference import predict_gnn_tg  # noqa: E402
except ImportError:
    GNN_AVAILABLE = False

PERMEABILITY_AVAILABLE = True
try:
    # pyrefly: ignore [missing-import]
    from permeability_to_D import permeability_to_D  # noqa: E402
except ImportError:
    PERMEABILITY_AVAILABLE = False

st.set_page_config(
    page_title="Polymer Property Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Known example polymers -- lets a user pick instead of typing.
# ---------------------------------------------------------------------------
EXAMPLE_POLYMERS = {
    "Polystyrene": "CC(c1ccccc1)C",
    "Polyethylene": "CC",
    "Polypropylene": "CC(C)",
    "PMMA": "CC(C)(C(=O)OC)",
    "PVC": "CC(Cl)",
}
CUSTOM_LABEL = "Custom / paste your own SMILES"

# ---------------------------------------------------------------------------
# Styling -- design tokens + component styles, injected once.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --primary: #818cf8;
        --primary-dark: #a5b4fc;
        --primary-light: #2a2a4a;
        --success: #34d399;
        --success-light: #0f2e24;
        --amber: #fbbf24;
        --amber-light: #3a2e0f;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --border: #2e3548;
        --surface: #171b2b;
        --bg: #0f1220;
        --radius: 12px;
        --shadow: 0 1px 2px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.3);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { background: var(--bg); }

    .block-container {
        max-width: 820px;
        padding-top: 1.75rem;
        padding-bottom: 2.25rem;
    }

    /* Tighten Streamlit's default vertical gaps between elements */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 0 !important;
    }
    div.element-container { margin-bottom: 0.35rem; }

    /* ---- Compact hero ---- */
    .ppp-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: var(--radius);
        padding: 1.15rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-md);
        color: white;
        flex-wrap: wrap;
    }
    .ppp-hero-text h1 {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0 0 0.15rem 0;
        color: white;
        letter-spacing: -0.02em;
    }
    .ppp-hero-text p {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.9);
        margin: 0;
    }
    .ppp-hero-badge {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: rgba(255,255,255,0.18);
        padding: 0.35rem 0.7rem;
        border-radius: 8px;
        white-space: nowrap;
    }

    /* ---- Card wrapper ---- */
    .ppp-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: var(--shadow);
    }
    .ppp-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .ppp-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.65rem 0 0.35rem 1px;
    }
    .ppp-label:first-of-type { margin-top: 0; }

    /* ---- Inputs: dark surface, subtle border, purple focus ---- */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        border-radius: 10px !important;
        border: 1.5px solid var(--border) !important;
        color: var(--text) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    .stTextInput input {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 0.75rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-light) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover { border-color: var(--primary) !important; }

    /* ---- Example chips ---- */
    div[data-testid="column"] .stButton button {
        background: var(--primary-light) !important;
        color: var(--primary-dark) !important;
        box-shadow: none !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 0.6rem !important;
        border: 1px solid #3d3d6b !important;
        border-radius: 8px !important;
        width: 100%;
        transition: background 0.15s ease, color 0.15s ease;
    }
    div[data-testid="column"] .stButton button:hover {
        background: var(--primary) !important;
        color: #0f1220 !important;
        transform: none;
    }

    /* ---- Segmented tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
        background: #1a1f33;
        padding: 0.25rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.86rem;
        color: var(--text-muted);
        padding: 0.45rem 0.9rem;
        transition: background 0.15s ease, color 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--primary-dark) !important;
        box-shadow: var(--shadow);
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.85rem; }

    /* ---- Primary buttons ---- */
    .stButton button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.25);
        transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
    }
    .stButton button:hover {
        background: var(--primary-dark);
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
        transform: translateY(-1px);
    }
    .stButton button:disabled {
        background: #2e3548 !important;
        color: #6b7280 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* ---- Validation states ---- */
    .ppp-valid {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--success);
        background: var(--success-light);
        border-radius: 7px;
        padding: 0.4rem 0.7rem;
        margin-top: 0.4rem;
        display: inline-block;
    }
    .ppp-invalid {
        font-size: 0.82rem;
        font-weight: 600;
        color: #f87171;
        background: #3a1616;
        border-radius: 7px;
        padding: 0.4rem 0.7rem;
        margin-top: 0.4rem;
        display: inline-block;
    }
    .ppp-hint-inline {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }

    /* ---- Result section ---- */
    .ppp-result-card {
        background: linear-gradient(135deg, #23234a 0%, #1c1f38 100%);
        border: 1px solid #3d3d6b;
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        text-align: center;
        animation: ppp-fade-in 0.35s ease;
    }
    .ppp-result-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--primary-dark);
        margin-bottom: 0.3rem;
    }
    .ppp-result-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .ppp-result-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }
    @keyframes ppp-fade-in {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .ppp-explain {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.55;
        background: var(--primary-light);
        border-left: 3px solid var(--primary);
        padding: 0.65rem 0.85rem;
        border-radius: 8px;
        margin-top: 0.7rem;
    }
    .ppp-explain b { color: var(--text); }

    /* ---- Tg scale ---- */
    .ppp-scale-wrap { margin-top: 0.9rem; }
    .ppp-scale-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .ppp-scale-track {
        position: relative;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, #93c5fd 0%, #a5b4fc 45%, #f9a8d4 100%);
    }
    .ppp-scale-marker {
        position: absolute;
        top: 50%;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--text);
        border: 3px solid var(--bg);
        box-shadow: 0 1px 6px rgba(0,0,0,0.55);
        transform: translate(-50%, -50%);
    }
    .ppp-scale-value {
        text-align: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.4rem;
    }

    /* ---- Molecule card ---- */
    .ppp-mol-card {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: var(--radius);
        padding: 0.75rem;
        text-align: center;
        height: 100%;
    }
    .ppp-mol-card img {
        border-radius: 8px;
        background: #ffffff;
        padding: 0.5rem;
    }
    .ppp-mol-caption {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        word-break: break-all;
    }

    /* ---- Model info ---- */
    .ppp-pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--primary-dark);
        background: var(--primary-light);
        border-radius: 9px;
        padding: 0.6rem 0.7rem;
        margin-bottom: 0.7rem;
    }
    .ppp-pipeline .arrow { color: var(--text-muted); font-weight: 400; }
    .ppp-info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
    }
    .ppp-info-item {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
    }
    .ppp-info-item .k {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .ppp-info-item .v {
        font-size: 0.83rem;
        font-weight: 600;
        color: var(--text);
        margin-top: 0.1rem;
    }

    /* ---- Footer ---- */
    .ppp-footer {
        text-align: center;
        font-size: 0.78rem;
        color: var(--text-muted);
        padding-top: 0.75rem;
        line-height: 1.5;
    }

    #MainMenu, footer, header {visibility: hidden;}

    @media (max-width: 640px) {
        .ppp-hero { flex-direction: column; align-items: flex-start; }
        .ppp-info-grid { grid-template-columns: 1fr 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend helpers -- unchanged logic, only wrapped for reuse/error-handling.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_tg_model():
    return joblib.load("models/xgboost_tg.joblib")


@st.cache_resource
def load_gnn_model():
    """Load the Phase 2 GNN. Caller must check os.path.exists first."""
    model = PolymerGNN(node_feature_dim=12, edge_feature_dim=5, hidden_dim=64)
    state = torch.load("models/gnn_tg.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_permeability_model():
    """Load the Phase 3 permeability model. Caller must check os.path.exists first."""
    return joblib.load("models/permeability_xgb.joblib")


def featurize_smiles(smiles, n_bits=256, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(generator.GetFingerprint(mol)).reshape(1, -1)


@st.cache_data(show_spinner=False)
def _cached_pinn_release_curve(D_rounded):
    """Train the PINN for a given (rounded) D and return the release curve.

    Cached so repeat clicks at the same slider position are instant instead
    of retraining every time. Uses the existing train_pinn() unchanged.
    """
    model = train_pinn(D=D_rounded, model_out=None, n_epochs=800,
                        n_lbfgs_rounds=5, verbose=False)
    model.eval()
    t_plot = np.linspace(0, 5.0, 200)
    x_center = torch.full((len(t_plot), 1), 0.5, dtype=torch.float32)
    t_tensor = torch.tensor(t_plot, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        C_center = model(x_center, t_tensor).numpy().flatten()
    return t_plot, C_center


def render_molecule_image(smiles, size=(260, 220)):
    """Best-effort 2D structure render. Returns None if it can't be drawn."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "smiles_value" not in st.session_state:
    st.session_state.smiles_value = EXAMPLE_POLYMERS["Polystyrene"]
if "polymer_choice" not in st.session_state:
    st.session_state.polymer_choice = "Polystyrene"


def _apply_dropdown_choice():
    choice = st.session_state.polymer_choice
    if choice != CUSTOM_LABEL:
        st.session_state.smiles_value = EXAMPLE_POLYMERS[choice]


def _use_example_chip(name, smiles):
    st.session_state.polymer_choice = name
    st.session_state.smiles_value = smiles


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ppp-hero">
        <div class="ppp-hero-text">
            <h1>🧪 Polymer Property Predictor</h1>
            <p>AI-powered prediction of polymer properties from molecular structure.</p>
        </div>
        <div class="ppp-hero-badge">🧪 Materials ML • XGBoost</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Prediction workspace -- polymer selection
# ---------------------------------------------------------------------------
st.markdown('<div class="ppp-card">', unsafe_allow_html=True)
st.markdown('<p class="ppp-card-title">🧬 Polymer selection</p>', unsafe_allow_html=True)

dropdown_options = list(EXAMPLE_POLYMERS.keys()) + [CUSTOM_LABEL]
st.selectbox(
    "Select polymer",
    dropdown_options,
    key="polymer_choice",
    on_change=_apply_dropdown_choice,
    label_visibility="collapsed",
)

st.markdown('<p class="ppp-label">Quick examples</p>', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLE_POLYMERS))
for col, (label, smiles) in zip(chip_cols, EXAMPLE_POLYMERS.items()):
    with col:
        st.button(label, key=f"example_{label}", on_click=_use_example_chip, args=(label, smiles))

st.markdown('<p class="ppp-label">SMILES</p>', unsafe_allow_html=True)
smiles_input = st.text_input(
    "Repeat-unit SMILES",
    key="smiles_value",
    help="The polymer's repeat-unit structure as a SMILES string.",
    label_visibility="collapsed",
    placeholder="e.g. CC(c1ccccc1)C",
)

_parsed_ok = bool(smiles_input) and Chem.MolFromSmiles(smiles_input) is not None
if smiles_input and _parsed_ok:
    st.markdown('<span class="ppp-valid">✓ Valid SMILES — ready to predict</span>', unsafe_allow_html=True)
elif smiles_input and not _parsed_ok:
    st.markdown(
        '<span class="ppp-invalid">⚠ Invalid SMILES — please enter a valid molecular structure</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<p class="ppp-hint-inline">Select an example or type a SMILES string above.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

polymer_name = st.session_state.polymer_choice if st.session_state.polymer_choice != CUSTOM_LABEL else "this polymer"

# ---------------------------------------------------------------------------
# Feature tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🌡 Glass Transition Temperature", "💊 Drug Release", "📖 About"])

with tab1:
    st.markdown('<div class="ppp-card">', unsafe_allow_html=True)
    predict_clicked = st.button(
        "🔮 Predict Glass Transition Temperature",
        key="tg_button",
        disabled=not _parsed_ok,
        width="stretch",
    )

    if predict_clicked:
        if not _parsed_ok:
            st.markdown(
                '<span class="ppp-invalid">⚠ Invalid SMILES — please enter a valid molecular structure before predicting</span>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("⏳ Analyzing molecular structure..."):
                try:
                    X = featurize_smiles(smiles_input)
                    if X is None:
                        raise ValueError("Could not parse this SMILES string.")
                    model = load_tg_model()
                    xgb_pred = float(model.predict(X)[0])

                    gnn_pred = None
                    if GNN_AVAILABLE and os.path.exists("models/gnn_tg.pt"):
                        try:
                            gnn_model = load_gnn_model()
                            gnn_pred = predict_gnn_tg(smiles_input, gnn_model)
                        except Exception:
                            gnn_pred = None

                    ensemble_pred = (xgb_pred + gnn_pred) / 2 if gnn_pred is not None else None

                    st.session_state["last_tg_result"] = {
                        "value": xgb_pred,
                        "gnn_value": gnn_pred,
                        "ensemble_value": ensemble_pred,
                        "smiles": smiles_input,
                        "name": polymer_name,
                    }
                except Exception:
                    st.session_state["last_tg_result"] = None
                    st.error("⚠ Something went wrong generating this prediction. Please try a different SMILES string.")

    result = st.session_state.get("last_tg_result")
    if result and result["smiles"] == smiles_input:
        pred = result["value"]

        # Map Tg onto a fixed, clearly-labeled visual scale. Purely a display
        # aid -- the underlying predicted value is used as-is everywhere else.
        scale_min, scale_max = -150.0, 250.0
        pct = max(0.0, min(1.0, (pred - scale_min) / (scale_max - scale_min))) * 100

        col_mol, col_result = st.columns([1, 1.3], gap="medium")
        with col_mol:
            st.markdown('<p class="ppp-label">Molecular structure</p>', unsafe_allow_html=True)
            img = render_molecule_image(result["smiles"])
            if img is not None:
                st.markdown('<div class="ppp-mol-card">', unsafe_allow_html=True)
                st.image(img, width="stretch")
                st.markdown(f'<div class="ppp-mol-caption">{result["smiles"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ppp-mol-card"><p class="ppp-hint-inline">Structure preview unavailable for this input.</p></div>',
                    unsafe_allow_html=True,
                )

        with col_result:
            st.markdown('<p class="ppp-label">Prediction result</p>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="ppp-result-card">
                    <div class="ppp-result-label">Glass Transition Temperature</div>
                    <div class="ppp-result-value">{pred:.1f} °C</div>
                    <div class="ppp-result-sub">{result['name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Multi-model comparison (Phase 1 XGBoost, Phase 2 GNN, ensemble) ----
        if result.get("gnn_value") is not None:
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("XGBoost (Phase 1)", f"{result['value']:.1f} °C")
            with metric_cols[1]:
                st.metric("GNN (Phase 2)", f"{result['gnn_value']:.1f} °C")
            with metric_cols[2]:
                st.metric("Ensemble average", f"{result['ensemble_value']:.1f} °C")
        else:
            st.info("ℹ️ The Phase 2 GNN model wasn't found, so only the XGBoost prediction is shown.")

        st.markdown(
            f"""
            <div class="ppp-explain">
            <b>What this means:</b> below the glass transition temperature,
            the polymer behaves more like a rigid glass. Above it, the
            polymer becomes softer and more rubber-like. Prediction from an
            XGBoost model trained on Morgan fingerprints (Phase 1).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="ppp-scale-wrap">
                <div class="ppp-scale-labels"><span>GLASSY</span><span>RUBBER-LIKE</span></div>
                <div class="ppp-scale-track">
                    <div class="ppp-scale-marker" style="left:{pct:.1f}%;"></div>
                </div>
                <div class="ppp-scale-value">Tg = {pred:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Model information (compact) ----
    st.markdown('<div class="ppp-card">', unsafe_allow_html=True)
    st.markdown('<p class="ppp-card-title">ℹ️ Model information</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ppp-pipeline">
            SMILES <span class="arrow">→</span> Molecular Fingerprint <span class="arrow">→</span> XGBoost <span class="arrow">→</span> Tg Prediction
        </div>
        <div class="ppp-info-grid">
            <div class="ppp-info-item"><div class="k">Model</div><div class="v">XGBoost</div></div>
            <div class="ppp-info-item"><div class="k">Input</div><div class="v">Polymer SMILES</div></div>
            <div class="ppp-info-item"><div class="k">Features</div><div class="v">Morgan Fingerprints</div></div>
            <div class="ppp-info-item"><div class="k">Output</div><div class="v">Glass Transition Temp.</div></div>
            <div class="ppp-info-item"><div class="k">Unit</div><div class="v">°C</div></div>
            <div class="ppp-info-item"><div class="k">Stage</div><div class="v">Phase 1</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="ppp-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ppp-explain" style="margin-top:0;">
        This trains a small physics-informed neural network live, using a
        diffusion coefficient placeholder. This demo uses a <b>reduced,
        faster</b> training run than our fully validated version &mdash;
        treat the curve's overall <b>shape</b> as illustrative, not precise.
        Training takes about 20&ndash;30 seconds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Structure-driven D (Phase 3 permeability model), if available ----
    auto_D = None
    if PERMEABILITY_AVAILABLE and os.path.exists("models/permeability_xgb.joblib") and _parsed_ok:
        try:
            perm_model = load_permeability_model()
            X_perm = featurize_smiles(smiles_input)
            if X_perm is not None:
                log10_permeability = float(perm_model.predict(X_perm)[0])
                auto_D = permeability_to_D(log10_permeability)
        except Exception:
            auto_D = None

    if auto_D is not None:
        st.info(
            f"🔬 Predicted CO₂ permeability suggests an effective diffusion "
            f"coefficient of **D ≈ {auto_D:.2f}**. This is derived from the "
            f"structure's predicted permeability as a physically-related "
            f"proxy — not a literally measured drug-diffusion coefficient."
        )

    use_structure_D = st.checkbox(
        "Use structure-predicted D",
        value=(auto_D is not None),
        disabled=(auto_D is None),
    )

    st.markdown('<p class="ppp-label">Effective diffusion coefficient (D)</p>', unsafe_allow_html=True)
    D_manual = st.slider(
        "Effective diffusion coefficient (D)",
        min_value=0.01, max_value=1.0, value=0.3, step=0.01,
        help="Higher D = faster release. In the full project this comes from a permeability model; here it's a direct slider so you can explore.",
        label_visibility="collapsed",
    )

    if use_structure_D and auto_D is not None:
        D_manual = auto_D

    release_speed = "slow release" if D_manual < 0.33 else "moderate release" if D_manual < 0.66 else "fast release"
    st.caption(f"Current value: **D = {D_manual:.2f}** — {release_speed}")

    run_clicked = st.button(
        "🚀 Train PINN and show release curve",
        key="pinn_button",
        disabled=not _parsed_ok,
        width="stretch",
    )
    if not _parsed_ok and not run_clicked:
        st.markdown(
            '<p class="ppp-hint-inline">Enter a valid SMILES string above to enable this button.</p>',
            unsafe_allow_html=True,
        )

    if run_clicked:
        try:
            D_key = round(D_manual, 2)
            with st.spinner("⏳ Training physics-informed model... (about 20-30 seconds)"):
                t_plot, C_center = _cached_pinn_release_curve(D_key)
            st.caption(f"Trained with D = {D_key:.2f} (this exact value determines the curve shape).")

            fig, ax = plt.subplots(figsize=(7, 4.2))
            fig.patch.set_facecolor("#171b2b")
            ax.set_facecolor("#171b2b")
            ax.plot(t_plot, C_center, linewidth=2.5, color="#818cf8")
            ax.fill_between(t_plot, C_center, alpha=0.15, color="#818cf8")
            ax.set_xlabel("Time", fontsize=10, color="#94a3b8")
            ax.set_ylabel("Concentration at slab center", fontsize=10, color="#94a3b8")
            ax.set_title(f"Predicted release curve: {polymer_name} (D={D_manual:.2f})",
                          fontsize=12, fontweight="bold", color="#e2e8f0")
            ax.tick_params(colors="#64748b")
            ax.grid(alpha=0.15, color="#64748b")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("#2e3548")
            st.pyplot(fig)
            st.markdown(
                """
                <div class="ppp-explain">
                <b>How to read this:</b> the curve shows how much drug
                concentration remains at the center of the polymer slab over
                time. A steep early drop means fast initial release; a curve
                that flattens out means release has mostly finished.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.error("⚠ The simulation could not complete. Please try again or adjust the diffusion coefficient.")

    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="ppp-card">', unsafe_allow_html=True)
    st.markdown('<p class="ppp-card-title">📖 About this project</p>', unsafe_allow_html=True)
    st.markdown(
        "A from-scratch ML project predicting polymer properties from "
        "molecular structure, built in three phases."
    )
    st.markdown(
        """
| Phase | Approach | Result |
|---|---|---|
| 1 | Random Forest / XGBoost on Morgan fingerprints | XGBoost MAE 27.5°C, R² 0.87 |
| 2 | Graph Neural Network (GINEConv) | Ensemble MAE 26.7°C, R² 0.855 |
| 3 | Physics-Informed Neural Network (diffusion PDE) | Mean error 0.0062 vs. exact solution |
"""
    )
    st.caption(
        "The drug-release D shown in the Drug Release tab is a "
        "permeability-derived proxy, not a literal drug-diffusion "
        "coefficient. See the GitHub README for full methodology and "
        "limitations."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="ppp-footer">
    Built as a learning project: Classical ML → Graph Neural Networks → Physics-Informed Neural Networks.<br>
    See the README for methodology and limitations.
    </div>
    """,
    unsafe_allow_html=True,
)
