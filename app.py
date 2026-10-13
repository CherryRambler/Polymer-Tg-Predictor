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
# pyrefly: ignore [missing-import]
import streamlit.components.v1 as components
import json
import numpy as np
import pandas as pd
import torch
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DRAW_AVAILABLE = True
try:
    from rdkit.Chem import Draw
except ImportError:
    DRAW_AVAILABLE = False
import matplotlib.pyplot as plt
import sys
import os
import html
import itertools

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "phase4_neural_operator"))

DEEPONET_AVAILABLE = True
try:
    # pyrefly: ignore [missing-import]
    from deeponet_model import DeepONet  # noqa: E402
except ImportError:
    DEEPONET_AVAILABLE = False

st.set_page_config(
    page_title="Polymer Intelligence Lab",
    page_icon="🧪",
    layout="wide",
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
# Design system ("Polymer Intelligence Lab") -- design tokens, one injected
# stylesheet and small HTML helpers, so the UI code below never repeats raw
# colors or markup. Pure presentation: nothing here touches models, features
# or predictions. Native widget colors/fonts come from .streamlit/config.toml
# (Streamlit only reads its theme from there); keep both in sync.
#
# Cards are native `st.container(border=True, key=...)` blocks; Streamlit adds
# an `st-key-<key>` class to them, which is what the CSS targets. That avoids
# opening an HTML <div> in one st.markdown() call and closing it in another
# (which never actually wraps the widgets in between).
# ---------------------------------------------------------------------------
TOKENS = {
    # Surfaces
    "bg": "#080B14",
    "bg-2": "#0D1220",
    "surface": "#111827",
    "surface-2": "#151D2E",
    "border": "#263449",
    "border-strong": "#33476A",
    # Accents
    "cyan": "#22D3EE",
    "violet": "#8B5CF6",
    "violet-soft": "#A78BFA",
    "amber": "#FBBF24",
    # Text
    "text": "#F1F5F9",
    "text-muted": "#94A3B8",
    "text-faint": "#64748B",
    # Status
    "success": "#34D399",
    "error": "#FB7185",
    # Shape
    "radius-sm": "8px",
    "radius": "12px",
    "radius-lg": "16px",
    # Type
    "font-sans": "'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
    "font-mono": "'Geist Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace",
}

# Chart series colors -- shared by every matplotlib figure in the app.
CURVE_PRIMARY = TOKENS["cyan"]
CURVE_SECONDARY = TOKENS["violet-soft"]

_CSS = """
/* ---------- Page canvas + full-viewport background layer ---------- */
/* The Quantum Swarm iframe (quantum_background) is pinned to the viewport
   behind everything; it ignores the pointer, so content stays fully usable. */
.stApp { background: var(--pil-bg); }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
    background: transparent;
}
.st-key-pil-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
}
.st-key-pil-bg iframe {
    position: fixed;
    inset: 0;
    width: 100vw !important;
    height: 100vh !important;
    height: 100dvh !important;
    border: 0;
    pointer-events: none;
}
/* Collapse the layout slot the background occupies in the page flow */
[data-testid="stElementContainer"]:has(> .st-key-pil-bg),
.stVerticalBlock > div:has(> .st-key-pil-bg) { margin: 0; height: 0; min-height: 0; }
[data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stMainBlockContainer"] {
    max-width: 1160px;
    padding: 1rem 1.5rem 3rem;
}
#MainMenu, footer, header { visibility: hidden; }
/* Text that sits straight on the animated background gets a soft halo */
.pil-on-bg { text-shadow: 0 1px 16px rgba(8, 11, 20, 0.9), 0 0 2px rgba(8, 11, 20, 0.8); }

/* ---------- Typography primitives ---------- */
.pil-mono { font-family: var(--pil-font-mono); }
.pil-eyebrow {
    font-family: var(--pil-font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pil-cyan);
}
.pil-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--pil-text-muted);
    margin: 0.2rem 0 0.1rem 1px;
}

/* ---------- Model status strip (under the hero) ---------- */
.pil-status {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.45rem;
    margin: 0.15rem 0 0.4rem;
}
.pil-status-item {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--pil-font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    color: var(--pil-text-muted);
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid var(--pil-border);
    border-radius: 999px;
    padding: 0.28rem 0.7rem 0.28rem 0.55rem;
}
.pil-status-item .dot {
    position: relative;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--pil-success);
    box-shadow: 0 0 8px rgba(52, 211, 153, 0.6);
}
.pil-status-item.ok .dot::after {
    content: "";
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    border: 1px solid rgba(52, 211, 153, 0.55);
    animation: pil-pulse 2.6s ease-out infinite;
}
.pil-status-item.off { color: var(--pil-text-faint); }
.pil-status-item.off .dot {
    background: transparent;
    border: 1px solid var(--pil-text-faint);
    box-shadow: none;
}
@keyframes pil-pulse {
    0% { transform: scale(0.6); opacity: 0.9; }
    100% { transform: scale(1.9); opacity: 0; }
}
/* ---------- Section headers ---------- */
.pil-section {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    margin: 0.35rem 0 0.2rem;
}
.pil-section-glyph {
    flex: 0 0 auto;
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: var(--pil-radius);
    border: 1px solid var(--pil-border);
    background:
        radial-gradient(circle at 30% 20%, rgba(34, 211, 238, 0.16), transparent 70%),
        var(--pil-surface);
}
.pil-section-title {
    font-size: 1.35rem;
    font-weight: 650;
    letter-spacing: -0.015em;
    color: var(--pil-text);
    line-height: 1.25;
    margin: 0.2rem 0 0.25rem;
}
.pil-section-desc {
    font-size: 0.9rem;
    color: var(--pil-text-muted);
    line-height: 1.55;
    max-width: 72ch;
    margin: 0;
}

/* ---------- Panels (native bordered containers) ----------
   Each section gets its own translucent glass panel -- just enough contrast
   to read text over the moving particles; never one card around the page. */
[class*="st-key-pil-card-"], [class*="st-key-pil-accent-"] {
    background: rgba(13, 18, 32, 0.68);
    -webkit-backdrop-filter: blur(10px) saturate(120%);
    backdrop-filter: blur(10px) saturate(120%);
    border-color: rgba(51, 71, 106, 0.7) !important;
    border-radius: var(--pil-radius-lg) !important;
    padding: 1.2rem 1.3rem !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 14px 32px -18px rgba(0, 0, 0, 0.7);
    transition: border-color 0.2s ease;
}
[class*="st-key-pil-card-"]:hover { border-color: var(--pil-border-strong) !important; }
/* Emphasized panel: thin luminous cyan->violet edge along the top */
[class*="st-key-pil-accent-"] {
    background:
        linear-gradient(90deg, rgba(34, 211, 238, 0.7), rgba(139, 92, 246, 0.7)) top / 100% 1px no-repeat,
        radial-gradient(600px 160px at 0% 0%, rgba(34, 211, 238, 0.06), transparent 70%),
        rgba(13, 18, 32, 0.74);
    border-color: var(--pil-border-strong) !important;
}
.pil-card-head {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 0.15rem;
}
.pil-card-head .node {
    flex: 0 0 auto;
    width: 8px;
    height: 8px;
    margin-top: 0.42rem;
    border-radius: 50%;
    background: var(--pil-cyan);
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15), 0 0 10px rgba(34, 211, 238, 0.5);
}
.pil-card-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--pil-text);
    line-height: 1.4;
}
.pil-card-desc {
    font-size: 0.82rem;
    color: var(--pil-text-muted);
    line-height: 1.5;
    margin-top: 0.1rem;
}

/* ---------- Buttons ---------- */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #22D3EE 0%, #38BDF8 100%);
    color: #04121A;
    border: 1px solid rgba(34, 211, 238, 0.9);
    font-weight: 600;
    min-height: 2.6rem;
    box-shadow: 0 6px 20px -8px rgba(34, 211, 238, 0.55);
    transition: box-shadow 0.18s ease, transform 0.18s ease, filter 0.18s ease;
}
[data-testid="stBaseButton-primary"]:hover {
    color: #04121A;
    border-color: #67E8F9;
    filter: brightness(1.06);
    box-shadow: 0 8px 26px -8px rgba(34, 211, 238, 0.75);
    transform: translateY(-1px);
}
[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-primary"] p { font-weight: 600; }
[data-testid="stBaseButton-primary"]:disabled {
    background: var(--pil-surface-2);
    color: var(--pil-text-faint);
    border-color: var(--pil-border);
    box-shadow: none;
    transform: none;
    filter: none;
}
[data-testid="stBaseButton-secondary"] {
    background: var(--pil-surface-2);
    border: 1px solid var(--pil-border);
    color: var(--pil-text);
    transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: rgba(34, 211, 238, 0.6);
    color: #A5F3FC;
    background: #17213A;
}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {
    outline: 2px solid var(--pil-cyan);
    outline-offset: 2px;
}

/* Example chips: compact pills; the current selection is rendered as primary */
.st-key-pil-chips button {
    min-height: 2rem;
    padding: 0.2rem 0.85rem;
    border-radius: 999px;
    box-shadow: none;
    transform: none;
}
.st-key-pil-chips button p { font-size: 0.8rem; }
.st-key-pil-chips [data-testid="stBaseButton-primary"],
.st-key-pil-chips [data-testid="stBaseButton-primary"]:hover {
    background: rgba(34, 211, 238, 0.12);
    color: #A5F3FC;
    border-color: rgba(34, 211, 238, 0.65);
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.08);
    transform: none;
    filter: none;
}

/* ---------- Inputs ---------- */
.stTextInput [data-baseweb="input"], .stTextArea [data-baseweb="textarea"],
.stSelectbox [data-baseweb="select"] > div {
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
.stTextInput [data-baseweb="input"]:focus-within,
.stTextArea [data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.16);
}
/* SMILES are chemistry, not prose: monospace wherever they are typed */
.st-key-smiles_value input, .st-key-pil-card-batch-input textarea {
    font-family: var(--pil-font-mono);
    font-size: 0.88rem;
}

/* ---------- Tabs: segmented navigation (ARIA roles: stable across versions) ---------- */
.stTabs [role="tablist"] {
    gap: 0.25rem;
    background: rgba(13, 18, 32, 0.85);
    border: 1px solid var(--pil-border);
    padding: 0.3rem;
    border-radius: var(--pil-radius);
    overflow-x: auto;
    scrollbar-width: none;
}
.stTabs [role="tablist"]::after { display: none; }
.stTabs [role="tab"] {
    height: auto;
    flex: 0 0 auto;
    border-radius: 9px;
    padding: 0.5rem 0.95rem;
    color: var(--pil-text-muted);
    transition: background 0.18s ease, color 0.18s ease;
}
.stTabs [role="tab"] p { font-size: 0.86rem; font-weight: 550; white-space: nowrap; }
.stTabs [role="tab"]:hover { color: var(--pil-text); background: rgba(21, 29, 46, 0.8); }
.stTabs [role="tab"][aria-selected="true"] {
    background: var(--pil-surface-2);
    color: #A5F3FC;
    box-shadow: inset 0 0 0 1px var(--pil-border-strong), inset 0 -2px 0 var(--pil-cyan);
}
.stTabs [role="tab"][aria-selected="true"] p { color: #A5F3FC; }
.stTabs [role="tab"]:focus-visible { outline: 2px solid var(--pil-cyan); outline-offset: 1px; }
.stTabs [role="tabpanel"] { padding-top: 1rem; }

/* ---------- Metrics, expanders, images ---------- */
[data-testid="stMetric"] {
    background: rgba(8, 11, 20, 0.55);
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius);
    padding: 0.7rem 0.9rem;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--pil-text-muted);
}
[data-testid="stMetricValue"] {
    font-family: var(--pil-font-mono);
    font-size: 1.45rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
[data-testid="stExpander"] details {
    border-color: var(--pil-border);
    border-radius: var(--pil-radius);
    background: rgba(17, 24, 39, 0.6);
}
.st-key-pil-architecture [data-testid="stImage"] img { border-radius: 10px; }
/* RDKit draws black-on-white PNGs. Display-only dark rendering: invert +
   hue-rotate keeps heteroatom colors (O stays red, N blue), and the screen
   blend lets the card surface show through the (now black) background. */
.st-key-pil-structure [data-testid="stImage"] {
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius);
    background:
        radial-gradient(circle at 50% 40%, rgba(34, 211, 238, 0.07), transparent 70%),
        var(--pil-bg-2);
    padding: 0.4rem;
}
.st-key-pil-structure [data-testid="stImage"] img {
    filter: invert(1) hue-rotate(180deg) brightness(1.05);
    mix-blend-mode: screen;
    max-height: 230px;
    object-fit: contain;
}

/* ---------- Validation badges ---------- */
.pil-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 550;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    border: 1px solid transparent;
}
.pil-badge.ok { color: var(--pil-success); background: rgba(52, 211, 153, 0.09); border-color: rgba(52, 211, 153, 0.28); }
.pil-badge.err { color: var(--pil-error); background: rgba(251, 113, 133, 0.09); border-color: rgba(251, 113, 133, 0.3); }
.pil-hint { font-size: 0.82rem; color: var(--pil-text-muted); margin: 0.1rem 0 0; }

/* ---------- Primary result ---------- */
.pil-result {
    position: relative;
    border-radius: var(--pil-radius-lg);
    border: 1px solid rgba(34, 211, 238, 0.35);
    background:
        radial-gradient(420px 200px at 15% 0%, rgba(34, 211, 238, 0.14), transparent 70%),
        radial-gradient(380px 200px at 100% 100%, rgba(139, 92, 246, 0.14), transparent 70%),
        var(--pil-bg-2);
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.05), 0 18px 40px -20px rgba(34, 211, 238, 0.35);
    padding: 1.2rem 1.35rem 1.1rem;
    animation: pil-rise 0.35s ease;
}
.pil-result-label { margin-bottom: 0.45rem; }
.pil-result-value {
    font-family: var(--pil-font-mono);
    font-size: 3rem;
    font-weight: 500;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: var(--pil-text);
    font-variant-numeric: tabular-nums;
}
.pil-result-value .unit { font-size: 1.35rem; color: var(--pil-cyan); margin-left: 0.3rem; letter-spacing: 0; }
.pil-result-sub { font-size: 0.83rem; color: var(--pil-text-muted); margin-top: 0.4rem; }
@keyframes pil-rise {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Tg scale (display aid only) */
.pil-scale { margin-top: 1.05rem; }
.pil-scale-labels, .pil-scale-ends {
    display: flex;
    justify-content: space-between;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    color: var(--pil-text-muted);
}
.pil-scale-labels { font-weight: 600; margin-bottom: 0.35rem; }
.pil-scale-ends { font-family: var(--pil-font-mono); letter-spacing: 0; margin-top: 0.4rem; color: var(--pil-text-faint); }
.pil-scale-track {
    position: relative;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22D3EE 0%, #8B5CF6 55%, #FBBF24 100%);
    opacity: 0.9;
}
.pil-scale-marker {
    position: absolute;
    top: 50%;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--pil-text);
    border: 3px solid var(--pil-bg);
    box-shadow: 0 0 0 1px rgba(241, 245, 249, 0.5), 0 0 12px rgba(241, 245, 249, 0.45);
    transform: translate(-50%, -50%);
}

/* ---------- Callouts, readouts, empty states ---------- */
.pil-callout {
    font-size: 0.86rem;
    color: var(--pil-text-muted);
    line-height: 1.6;
    background: linear-gradient(90deg, rgba(34, 211, 238, 0.07), rgba(139, 92, 246, 0.03));
    border: 1px solid var(--pil-border);
    border-left: 2px solid var(--pil-cyan);
    padding: 0.7rem 0.95rem;
    border-radius: 10px;
}
.pil-callout b { color: var(--pil-text); font-weight: 600; }
.pil-readout {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--pil-text-muted);
}
.pil-readout .v { font-family: var(--pil-font-mono); font-size: 1.15rem; color: var(--pil-text); }
.pil-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 550;
    color: #C4B5FD;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 999px;
    padding: 0.12rem 0.6rem;
}
.pil-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    text-align: center;
    min-height: 220px;
    padding: 1.5rem;
    border: 1px dashed var(--pil-border-strong);
    border-radius: var(--pil-radius);
    background: rgba(8, 11, 20, 0.5);
    color: var(--pil-text-muted);
    font-size: 0.86rem;
    line-height: 1.55;
}
.pil-empty.compact { min-height: 0; padding: 1rem; }
.pil-empty p { margin: 0; max-width: 40ch; }

/* ---------- Model information ---------- */
.pil-pipeline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.35rem 0;
    margin: 0.3rem 0 0.2rem;
}
.pil-pipeline .step {
    font-size: 0.78rem;
    font-weight: 550;
    color: var(--pil-text);
    background: var(--pil-surface-2);
    border: 1px solid var(--pil-border);
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
}
.pil-pipeline .step:last-child { border-color: rgba(34, 211, 238, 0.5); color: #A5F3FC; }
.pil-pipeline .link {
    width: 22px;
    height: 1px;
    background: linear-gradient(90deg, rgba(34, 211, 238, 0.7), rgba(139, 92, 246, 0.7));
}
.pil-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.55rem;
}
.pil-grid .item {
    background: rgba(8, 11, 20, 0.5);
    border: 1px solid var(--pil-border);
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
}
.pil-grid .k {
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--pil-text-muted);
}
.pil-grid .v { font-size: 0.85rem; font-weight: 550; color: var(--pil-text); margin-top: 0.15rem; }
.pil-grid .v.mono { font-family: var(--pil-font-mono); font-weight: 400; }

/* ---------- About: phase cards ---------- */
.pil-phases {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.75rem;
}
.pil-phase {
    background: rgba(13, 18, 32, 0.68);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius-lg);
    padding: 1rem 1.1rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.pil-phase:hover { border-color: rgba(34, 211, 238, 0.45); transform: translateY(-1px); }
.pil-phase .num {
    display: inline-grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    font-family: var(--pil-font-mono);
    font-size: 0.78rem;
    color: #A5F3FC;
    background: rgba(34, 211, 238, 0.1);
    border: 1px solid rgba(34, 211, 238, 0.35);
}
.pil-phase .name { font-size: 0.95rem; font-weight: 600; color: var(--pil-text); margin: 0.6rem 0 0.25rem; }
.pil-phase .approach { font-size: 0.82rem; color: var(--pil-text-muted); line-height: 1.5; min-height: 2.5em; }
.pil-phase .result {
    font-family: var(--pil-font-mono);
    font-size: 0.78rem;
    color: var(--pil-text);
    border-top: 1px solid var(--pil-border);
    margin-top: 0.75rem;
    padding-top: 0.6rem;
}
.pil-list { margin: 0.2rem 0 0; padding-left: 1.1rem; color: var(--pil-text-muted); font-size: 0.86rem; line-height: 1.6; }
.pil-list li { margin-bottom: 0.3rem; }
.pil-list li::marker { color: var(--pil-cyan); }
.pil-list b { color: var(--pil-text); font-weight: 600; }
.pil-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.pil-tags .t {
    font-family: var(--pil-font-mono);
    font-size: 0.75rem;
    color: var(--pil-text);
    background: var(--pil-surface-2);
    border: 1px solid var(--pil-border);
    border-radius: 7px;
    padding: 0.22rem 0.6rem;
}
.pil-mol-caption {
    font-family: var(--pil-font-mono);
    font-size: 0.75rem;
    color: var(--pil-text-muted);
    text-align: center;
    word-break: break-all;
    margin-top: 0.1rem;
}

/* ---------- Top navigation (three-step flow) ---------- */
.st-key-pil-nav {
    align-items: center;
    justify-content: space-between;
    padding: 0.35rem 0 0.9rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(38, 52, 73, 0.6);
}
.pil-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    font-family: var(--pil-font-mono);
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pil-text);
    white-space: nowrap;
}
.pil-brand .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--pil-cyan);
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.8);
}
.st-key-pil-steps button {
    min-height: 2.1rem;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    box-shadow: none;
    transform: none;
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
}
.st-key-pil-steps [data-testid="stBaseButton-secondary"] { background: rgba(17, 24, 39, 0.6); }
.st-key-pil-steps button p { font-family: var(--pil-font-mono); font-size: 0.76rem; letter-spacing: 0.04em; }
.st-key-pil-steps [data-testid="stBaseButton-primary"],
.st-key-pil-steps [data-testid="stBaseButton-primary"]:hover {
    background: rgba(34, 211, 238, 0.14);
    color: #A5F3FC;
    border-color: rgba(34, 211, 238, 0.7);
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.08);
    transform: none;
    filter: none;
}
.st-key-pil-steps [data-testid="stBaseButton-primary"] p { color: #A5F3FC; }

/* ---------- Page heading (Calculations / Results) ---------- */
.pil-page-head { margin: 0.6rem 0 0.9rem; }
.pil-page-title {
    font-size: clamp(1.6rem, 3.4vw, 2.3rem);
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.15;
    color: var(--pil-text);
    margin: 0.35rem 0 0.4rem;
}
.pil-page-desc { font-size: 0.95rem; color: #CBD5E1; line-height: 1.6; max-width: 70ch; margin: 0; }

/* ---------- Overview hero ---------- */
.pil-hero {
    text-align: center;
    padding: clamp(2.5rem, 9vh, 6rem) 0 clamp(1.5rem, 4vh, 2.5rem);
}
.pil-hero h1 {
    margin: 0.9rem 0 1rem;
    padding: 0;
    font-family: var(--pil-font-mono);
    font-size: clamp(2.6rem, 9vw, 6rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.05em;
    text-transform: uppercase;
    color: transparent;
    background: linear-gradient(135deg, #FFFFFF, #CFFAFE 50%, #C4B5FD);
    -webkit-background-clip: text;
    background-clip: text;
    filter: drop-shadow(0 0 24px rgba(34, 211, 238, 0.25));
    text-shadow: none;
}
.pil-hero .pil-eyebrow { font-size: 0.76rem; }
.pil-hero-sub {
    display: inline-block;
    max-width: 40rem;
    margin: 0;
    padding: 0.45rem 1.1rem;
    font-size: 0.98rem;
    line-height: 1.55;
    color: #CBD5E1;
    background: rgba(8, 11, 20, 0.6);
    border: 1px solid rgba(38, 52, 73, 0.8);
    border-radius: 999px;
    -webkit-backdrop-filter: blur(6px);
    backdrop-filter: blur(6px);
}
.pil-intro {
    max-width: 68ch;
    margin: 1.6rem auto 0;
    text-align: center;
    font-size: 1rem;
    line-height: 1.7;
    color: #CBD5E1;
}
.pil-intro b { color: var(--pil-text); font-weight: 600; }
/* Start CTA: centered, larger than regular primary buttons */
.st-key-pil-cta { align-items: center; margin: 1.6rem 0 0.4rem; }
.st-key-pil-cta button { min-width: 15rem; min-height: 3.1rem; border-radius: 999px; }
.st-key-pil-cta button p { font-size: 1rem; letter-spacing: 0.02em; }

/* Section label straight on the background */
.pil-block-title {
    font-size: 1.15rem;
    font-weight: 650;
    letter-spacing: -0.01em;
    color: var(--pil-text);
    margin: 2.2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.pil-block-title::before {
    content: "";
    width: 18px;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--pil-cyan), var(--pil-violet));
}

/* Objectives / key features: open grids of small glass tiles */
.pil-feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
}
.pil-feature {
    background: rgba(13, 18, 32, 0.62);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(51, 71, 106, 0.6);
    border-radius: var(--pil-radius-lg);
    padding: 1rem 1.1rem;
}
.pil-feature .icon { font-size: 1.25rem; line-height: 1; }
.pil-feature .name { font-size: 0.95rem; font-weight: 600; color: var(--pil-text); margin: 0.55rem 0 0.3rem; }
.pil-feature .desc { font-size: 0.84rem; color: var(--pil-text-muted); line-height: 1.55; }
.pil-objectives { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.55rem;
                  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.pil-objectives li {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    font-size: 0.9rem;
    line-height: 1.55;
    color: #CBD5E1;
    background: rgba(8, 11, 20, 0.5);
    border: 1px solid rgba(38, 52, 73, 0.7);
    border-radius: var(--pil-radius);
    padding: 0.7rem 0.9rem;
}
.pil-objectives .n { font-family: var(--pil-font-mono); font-size: 0.78rem; color: var(--pil-cyan); padding-top: 0.12rem; }
.pil-objectives b { color: var(--pil-text); font-weight: 600; }

/* ---------- Results: run summary strip ---------- */
.pil-run {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.4rem;
    align-items: baseline;
    font-size: 0.85rem;
    color: var(--pil-text-muted);
    padding: 0.7rem 1rem;
    background: rgba(8, 11, 20, 0.55);
    border: 1px solid rgba(38, 52, 73, 0.8);
    border-radius: var(--pil-radius);
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
}
.pil-run .v { color: var(--pil-text); font-weight: 600; }
.pil-run .mono { font-family: var(--pil-font-mono); font-weight: 400; word-break: break-all; }

/* ---------- Footer ---------- */
.pil-footer {
    text-align: center;
    font-size: 0.78rem;
    color: var(--pil-text-faint);
    line-height: 1.6;
    border-top: 1px solid var(--pil-border);
    padding-top: 1.1rem;
    margin-top: 1rem;
}

/* ---------- Responsive + motion preferences ---------- */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 0.75rem 1rem 2.5rem; }
    .pil-section-glyph { display: none; }
    .pil-section-title { font-size: 1.15rem; }
    .pil-result-value { font-size: 2.4rem; }
    [class*="st-key-pil-card-"], [class*="st-key-pil-accent-"] { padding: 1rem !important; }
    .stTabs [role="tab"] { padding: 0.45rem 0.7rem; }
    .st-key-pil-nav { justify-content: center; }
    .st-key-pil-steps button { padding: 0.2rem 0.65rem; }
    .pil-hero-sub { border-radius: var(--pil-radius); }
    .pil-objectives { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
    .pil-status-item.ok .dot::after { animation: none; display: none; }
    .pil-result { animation: none; }
    .pil-phase, .pil-phase:hover, [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"]:hover { transition: none; transform: none; }
}
"""

_glyph_ids = itertools.count()


def _glyph_svg():
    """Polymer-chain glyph used in section headers: four units, bonded.
    Each copy gets its own gradient id -- a shared id would resolve to the
    first copy, which may sit inside a hidden tab and then fails to paint."""
    gid = f"pil-chain-{next(_glyph_ids)}"
    return (
        '<svg viewBox="0 0 40 40" width="30" height="30" aria-hidden="true">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#22D3EE"/><stop offset="1" stop-color="#8B5CF6"/>'
        "</linearGradient></defs>"
        f'<path d="M5 27 L14 13 L24 27 L34 13" fill="none" stroke="url(#{gid})" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="5" cy="27" r="2.6" fill="#22D3EE"/>'
        '<circle cx="14" cy="13" r="2.6" fill="#5EB8F3"/>'
        '<circle cx="24" cy="27" r="2.6" fill="#7C8CF4"/>'
        '<circle cx="34" cy="13" r="2.6" fill="#8B5CF6"/>'
        "</svg>"
    )


def _md(markup):
    st.markdown(markup, unsafe_allow_html=True)


def esc(text):
    """HTML-escape user-provided text (SMILES, names) before embedding it."""
    return html.escape(str(text))


def inject_css():
    """Inject the design tokens + stylesheet. Call once, right after set_page_config."""
    root = ":root{" + "".join(f"--pil-{k}:{v};" for k, v in TOKENS.items()) + "}"
    # st.html with only a <style> block adds no visible element / spacing.
    st.html(f"<style>{root}{_CSS}</style>")


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
def card(key, accent=False):
    """A bordered card for widgets. `accent=True` adds the luminous top edge;
    use it for the one card per section that holds the main result."""
    prefix = "pil-accent-" if accent else "pil-card-"
    return st.container(border=True, key=prefix + key)


def section_header(eyebrow, title, description=None):
    desc = f'<p class="pil-section-desc">{description}</p>' if description else ""
    _md(
        '<div class="pil-section">'
        f'<div class="pil-section-glyph">{_glyph_svg()}</div>'
        "<div>"
        f'<div class="pil-eyebrow">{eyebrow}</div>'
        f'<div class="pil-section-title" role="heading" aria-level="2">{title}</div>'
        f"{desc}"
        "</div></div>"
    )


def card_header(title, description=None):
    desc = f'<div class="pil-card-desc">{description}</div>' if description else ""
    _md(
        '<div class="pil-card-head"><span class="node"></span><div>'
        f'<div class="pil-card-title" role="heading" aria-level="3">{title}</div>{desc}'
        "</div></div>"
    )


def field_label(text):
    _md(f'<div class="pil-label">{text}</div>')


def hint(text):
    _md(f'<p class="pil-hint">{text}</p>')


def badge(text, ok=True):
    _md(f'<span class="pil-badge {"ok" if ok else "err"}">{text}</span>')


def callout(body_html):
    """Explanatory note. `body_html` is trusted, static markup."""
    _md(f'<div class="pil-callout">{body_html}</div>')


def empty_state(text, compact=False):
    _md(
        f'<div class="pil-empty{" compact" if compact else ""}">'
        f"{_glyph_svg()}<p>{text}</p></div>"
    )


def readout(name, value, note=None):
    note_html = f'<span class="pil-tag">{note}</span>' if note else ""
    _md(
        f'<div class="pil-readout"><span>Current value</span>'
        f'<span class="v">{name} = {value}</span>{note_html}</div>'
    )


def info_grid(pairs, mono_keys=()):
    """Compact key/value tiles. Values whose key is in `mono_keys` render in
    monospace (identifiers, parameters, file names)."""
    items = "".join(
        f'<div class="item"><div class="k">{k}</div>'
        f'<div class="v{" mono" if k in mono_keys else ""}">{v}</div></div>'
        for k, v in pairs
    )
    _md(f'<div class="pil-grid">{items}</div>')


def pipeline(steps):
    parts = '<span class="link"></span>'.join(f'<span class="step">{s}</span>' for s in steps)
    _md(f'<div class="pil-pipeline" aria-label="Model pipeline">{parts}</div>')


def tags(items):
    _md('<div class="pil-tags">' + "".join(f'<span class="t">{t}</span>' for t in items) + "</div>")


def bullet_list(items_html):
    _md('<ul class="pil-list">' + "".join(f"<li>{i}</li>" for i in items_html) + "</ul>")


# ---------------------------------------------------------------------------
# App-specific pieces
# ---------------------------------------------------------------------------
def status_bar(items):
    """Model availability strip. `items` = [(label, available, tooltip), ...]."""
    chips = "".join(
        f'<span class="pil-status-item {"ok" if ok else "off"}" title="{esc(tip)}">'
        f'<span class="dot"></span>{name}</span>'
        for name, ok, tip in items
    )
    _md(f'<div class="pil-status" role="list" aria-label="Model availability">{chips}</div>')


def result_card(label_text, value, unit, sub, scale=None):
    """The primary prediction readout. `scale` = (pct, min, max) adds the
    glassy -> rubber-like position bar (a display aid only)."""
    scale_html = ""
    if scale is not None:
        pct, lo, hi = scale
        scale_html = (
            '<div class="pil-scale">'
            '<div class="pil-scale-labels"><span>GLASSY</span><span>RUBBER-LIKE</span></div>'
            f'<div class="pil-scale-track"><div class="pil-scale-marker" style="left:{pct:.1f}%;"></div></div>'
            f'<div class="pil-scale-ends"><span>{lo:.0f} {unit}</span><span>{hi:.0f} {unit}</span></div>'
            "</div>"
        )
    _md(
        '<div class="pil-result" role="status">'
        f'<div class="pil-eyebrow pil-result-label">{label_text}</div>'
        f'<div class="pil-result-value">{value}<span class="unit">{unit}</span></div>'
        f'<div class="pil-result-sub">{sub}</div>'
        f"{scale_html}</div>"
    )


def mol_caption(smiles):
    _md(f'<div class="pil-mol-caption">{esc(smiles)}</div>')


def phase_grid(phases):
    """Overview page: one card per project phase. `phases` = [(num, name, approach, result), ...]."""
    cards = "".join(
        '<div class="pil-phase">'
        f'<span class="num">{num}</span>'
        f'<div class="name">{name}</div>'
        f'<div class="approach">{approach}</div>'
        f'<div class="result">{result}</div>'
        "</div>"
        for num, name, approach, result in phases
    )
    _md(f'<div class="pil-phases">{cards}</div>')


def footer(body_html):
    _md(f'<div class="pil-footer">{body_html}</div>')


def style_release_axes(fig, ax, title, legend=False):
    """Apply the dark lab styling to a release-curve matplotlib figure."""
    t = TOKENS
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlabel("Time (model units)", fontsize=10, color=t["text-muted"])
    ax.set_ylabel("Concentration at slab center, C / C₀", fontsize=10, color=t["text-muted"])
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=t["text"], loc="left", pad=10)
    ax.tick_params(colors=t["text-muted"], labelsize=9)
    ax.grid(alpha=0.35, color=t["border"], linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(t["border"])
    if legend:
        ax.legend(fontsize=9, facecolor=t["bg-2"], edgecolor=t["border"], labelcolor=t["text"])
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Quantum Swarm background -- the interactive particle "galaxy" that fills the
# whole viewport behind every page. Plain HTML/CSS/JS in a components.html
# iframe, which the CSS in _CSS (.st-key-pil-bg) pins to the viewport behind
# the app content with pointer-events disabled, so it never blocks clicks,
# typing or scrolling. Pointer input is read from the parent page instead
# (Streamlit component iframes are same-origin), which keeps the repulsion and
# click shockwaves working across the whole page. Physics: a golden-angle
# spiral of particles on springs. Always animated (by design, it ignores the
# OS reduced-motion setting); it only idles while the browser tab is hidden.
# ---------------------------------------------------------------------------
_QUANTUM_SWARM_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
html, body { margin: 0; height: 100%; overflow: hidden; background: #080B14; }
canvas { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
</style></head>
<body>
<canvas id="canvas" aria-hidden="true"></canvas>
<script>
(() => {
  const P = __PARAMS__;
  const CYAN = [34, 211, 238], VIOLET = [139, 92, 246], BG = "#080B14";
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(", ");

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { alpha: false });

  let needsFrame = true;   // paint one still frame even when paused (first paint, resize)
  let time = 0;
  let particles = [];
  const dim = { width: 0, height: 0, cx: 0, cy: 0 };
  const pointer = { x: -2000, y: -2000, isDown: false, radius: 150, shockwaves: [] };

  // Golden-angle spiral (sunflower), denser toward the center. Sized to reach
  // the viewport corners; particle count scales with screen area.
  function buildSwarm() {
    const { width, height, cx, cy } = dim;
    if (width === 0 || height === 0) return;
    const angleIncrement = Math.PI * 2 * ((1 + Math.sqrt(5)) / 2);
    const maxRadius = Math.hypot(width, height) * 0.52;
    const n = Math.round(Math.min(P.maxParticles, Math.max(P.minParticles, (width * height) / 1500)));
    particles = [];
    for (let i = 0; i < n; i++) {
      const t = Math.pow(i / (n - 1), 0.6);
      const dst = t * maxRadius;
      const angle = i * angleIncrement;
      particles.push({
        x: cx + Math.cos(angle) * dst, y: cy + Math.sin(angle) * dst, vx: 0, vy: 0,
        baseX: 0, baseY: 0, angle: angle, distance: dst,
        size: Math.random() * 1.5 + 0.5, excitation: 0,
        color: mix(CYAN, VIOLET, t),   // cyan core -> violet rim
      });
    }
  }

  function resize() {
    const width = window.innerWidth, height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);  // cap DPR for performance
    Object.assign(dim, { width: width, height: height, cx: width / 2, cy: height / 2 });
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildSwarm();
    needsFrame = true;
  }
  window.addEventListener("resize", resize);
  resize();

  function loop() {
    const animate = !document.hidden;  // always moving; only idles in a hidden browser tab
    if (!animate && !needsFrame) { requestAnimationFrame(loop); return; }
    needsFrame = false;
    if (animate) time += 0.002;  // global rotation speed
    const { width, height, cx, cy } = dim;
    const cyan = CYAN.join(", "), violet = VIOLET.join(", ");

    // 1. Background: deep navy with soft cyan / violet nebulae
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);
    const glowCyan = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(width, height) * 0.45);
    glowCyan.addColorStop(0, `rgba(${cyan}, 0.10)`);
    glowCyan.addColorStop(1, `rgba(${cyan}, 0)`);
    ctx.fillStyle = glowCyan;
    ctx.fillRect(0, 0, width, height);
    const glowViolet = ctx.createRadialGradient(width, height, 0, width, height, width * 0.6);
    glowViolet.addColorStop(0, `rgba(${violet}, 0.14)`);
    glowViolet.addColorStop(1, `rgba(${violet}, 0)`);
    ctx.fillStyle = glowViolet;
    ctx.fillRect(0, 0, width, height);

    // 2. Shockwaves expand and fade
    for (let s = pointer.shockwaves.length - 1; s >= 0; s--) {
      const sw = pointer.shockwaves[s];
      sw.radius += 15;
      sw.strength *= 0.92;
      if (sw.radius > sw.maxRadius || sw.strength < 0.01) pointer.shockwaves.splice(s, 1);
    }

    // 3. Physics: spring to the rotating orbit, pointer repulsion, shockwave impulses
    for (const p of particles) {
      const currentAngle = p.angle + time * (1 + 100 / (p.distance + 100));  // inner particles rotate faster
      p.baseX = cx + Math.cos(currentAngle) * p.distance;
      p.baseY = cy + Math.sin(currentAngle) * p.distance;
      p.vx += (p.baseX - p.x) * 0.02;
      p.vy += (p.baseY - p.y) * 0.02;

      const dxP = p.x - pointer.x, dyP = p.y - pointer.y;
      const distP = Math.sqrt(dxP * dxP + dyP * dyP);
      if (distP < pointer.radius && distP > 0) {
        const force = (pointer.radius - distP) / pointer.radius;
        const direction = pointer.isDown ? -0.5 : 1.5;  // pressed: pull in slightly
        p.vx += (dxP / distP) * force * direction;
        p.vy += (dyP / distP) * force * direction;
        p.excitation = Math.max(p.excitation, force);
      }

      for (const sw of pointer.shockwaves) {
        const dxS = p.x - sw.x, dyS = p.y - sw.y;
        const distS = Math.sqrt(dxS * dxS + dyS * dyS);
        const ringDelta = Math.abs(distS - sw.radius);
        if (ringDelta < 30 && distS > 0) {
          const impulse = (1 - ringDelta / 30) * sw.strength * 15;
          p.vx += (dxS / distS) * impulse;
          p.vy += (dyS / distS) * impulse;
          p.excitation = Math.max(p.excitation, 1.0);
        }
      }

      p.vx *= 0.88;  // damping
      p.vy *= 0.88;
      p.x += p.vx;
      p.y += p.vy;
      p.excitation *= 0.95;
    }

    // 4. Constellation lines between near neighbours (limited look-ahead keeps it cheap)
    ctx.lineWidth = 0.6;
    for (let i = 0; i < particles.length; i++) {
      const p1 = particles[i];
      const limit = Math.min(particles.length, i + 15);
      for (let j = i + 1; j < limit; j++) {
        const p2 = particles[j];
        const dx = p1.x - p2.x, dy = p1.y - p2.y;
        const distSq = dx * dx + dy * dy;
        if (distSq < 3600) {
          const opacity = 1 - Math.sqrt(distSq) / 60;
          const excitation = Math.max(p1.excitation, p2.excitation);
          const alpha = Math.min(1, opacity * 0.2 + excitation * 0.5);
          ctx.strokeStyle = `rgba(${cyan}, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }

    // 5. Particles, with a halo when excited
    for (const p of particles) {
      const radius = p.size + p.excitation * 2.5;
      ctx.fillStyle = `rgb(${p.color})`;
      if (p.excitation > 0.3) {
        ctx.save();
        ctx.globalAlpha = p.excitation * 0.4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius * 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(loop);
  }

  // Pointer input comes from the app page: this iframe sits behind the content
  // with pointer-events: none. It is pinned at (0, 0) and fills the viewport,
  // so the page's client coordinates are canvas coordinates.
  let host = null;
  try { host = window.parent.document; } catch (err) { host = null; }  // cross-origin: no interaction
  if (host) {
    host.addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, { passive: true });
    host.addEventListener("pointerdown", (e) => {
      pointer.isDown = true;
      pointer.x = e.clientX; pointer.y = e.clientY;
      pointer.shockwaves.push({ x: e.clientX, y: e.clientY, radius: 10, maxRadius: 200, strength: 0.8 });
    }, { passive: true });
    const release = () => { pointer.isDown = false; };
    host.addEventListener("pointerup", release, { passive: true });
    host.addEventListener("pointercancel", release, { passive: true });
    host.documentElement.addEventListener("pointerleave", () => {
      pointer.x = -2000; pointer.y = -2000; pointer.isDown = false;
    }, { passive: true });
  }

  requestAnimationFrame(loop);
})();
</script>
</body></html>
"""


def quantum_background(min_particles=220, max_particles=950):
    """Render the full-viewport Quantum Swarm background. Call it once, at the
    same point on every run (before any page content), so Streamlit keeps the
    same iframe -- and a running animation -- when the user switches pages."""
    params = json.dumps({"minParticles": min_particles, "maxParticles": max_particles})
    markup = _QUANTUM_SWARM_HTML.replace("__PARAMS__", params)
    with st.container(key="pil-bg"):
        if hasattr(st, "iframe"):
            st.iframe(markup, height=1)  # size comes from the .st-key-pil-bg CSS
        else:  # older Streamlit without st.iframe
            components.html(markup, height=0)


inject_css()



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


@st.cache_resource
def load_deeponet_model():
    """Load the Phase 4 DeepONet. Caller must check os.path.exists first."""
    model = DeepONet(hidden_dim=64, n_hidden_layers=3, latent_dim=64)
    state = torch.load("models/deeponet_diffusion.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_structure_driven_D(smiles, parsed_ok):
    """Predict an effective diffusion coefficient D from a SMILES string,
    via the Phase 3 permeability model + permeability_to_D(), if both are
    available. Returns None if the permeability pipeline isn't available,
    the SMILES doesn't parse, or anything goes wrong -- callers should
    treat None as "fall back to manual D".

    Shared by the PINN and DeepONet release curves so they always agree on the same structure-to-D value
    for a given polymer.
    """
    if not (PERMEABILITY_AVAILABLE and os.path.exists("models/permeability_xgb.joblib") and parsed_ok):
        return None
    try:
        perm_model = load_permeability_model()
        X_perm = featurize_smiles(smiles)
        if X_perm is None:
            return None
        log10_permeability = float(perm_model.predict(X_perm)[0])
        return permeability_to_D(log10_permeability)
    except Exception:
        return None


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


def deeponet_release_curve(D_value):
    """Evaluate the trained DeepONet for a given D. A single forward pass
    (no training loop), so this is fast enough to call directly on every
    slider movement -- the entire point of Phase 4.
    """
    model = load_deeponet_model()
    t_plot = np.linspace(0, 5.0, 200)
    D_tensor = torch.full((len(t_plot), 1), float(D_value), dtype=torch.float32)
    x_center = torch.full((len(t_plot), 1), 0.5, dtype=torch.float32)
    t_tensor = torch.tensor(t_plot, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        C_center = model(D_tensor, x_center, t_tensor).numpy().flatten()
    return t_plot, C_center


def load_deeponet_eval_results():
    """Read the real held-out MAE/R2 saved by deeponet_evaluate.py, if it
    has been run. Returns None if the results file doesn't exist yet --
    callers must not fabricate a number in that case.
    """
    path = os.path.join(os.path.dirname(__file__), "results", "deeponet_evaluation.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


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
# Session state + page flow
# ---------------------------------------------------------------------------
# Three pages, switched through session state (single script, no
# st.navigation): Overview -> Calculations -> Results. The page is mirrored to
# the ?page= query param so a refresh stays on the same page.
PAGES = {
    "overview": "01 · Overview",
    "calculate": "02 · Calculations",
    "results": "03 · Results",
}

if "page" not in st.session_state:
    _qp_page = st.query_params.get("page")
    st.session_state.page = _qp_page if _qp_page in PAGES else "overview"
if "smiles_value" not in st.session_state:
    st.session_state.smiles_value = EXAMPLE_POLYMERS["Polystyrene"]
if "polymer_choice" not in st.session_state:
    st.session_state.polymer_choice = "Polystyrene"

deeponet_ready = DEEPONET_AVAILABLE and os.path.exists("models/deeponet_diffusion.pt")

# Widget defaults live in session state (not in value=...), so the inputs keep
# their values across pages without Streamlit's "default value + Session State"
# warning. use_structure_D is seeded on the Calculations page, where D is known.
st.session_state.setdefault("D_slider", 0.3)
st.session_state.setdefault("train_pinn", not deeponet_ready)
st.session_state.setdefault("batch_text", "CC(c1ccccc1)C\nCC\nCC(C)(C(=O)OC)C")

# Streamlit drops a widget's session-state value on any run where that widget
# isn't rendered. The calculation inputs only render on the Calculations page,
# so re-assign them every run to keep them while the user is on other pages.
for _key in ("smiles_value", "polymer_choice", "use_structure_D", "D_slider",
             "train_pinn", "batch_text"):
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]


def go_to(page):
    st.session_state.page = page
    st.query_params["page"] = page


def _apply_dropdown_choice():
    choice = st.session_state.polymer_choice
    if choice != CUSTOM_LABEL:
        st.session_state.smiles_value = EXAMPLE_POLYMERS[choice]


def _use_example_chip(name, smiles):
    st.session_state.polymer_choice = name
    st.session_state.smiles_value = smiles


# ---------------------------------------------------------------------------
# Full-page background + top navigation (shared by all three pages)
# ---------------------------------------------------------------------------
# Rendered first, at the same spot every run, so the same background iframe
# (and its running animation) survives page switches.
quantum_background()

with st.container(horizontal=True, key="pil-nav", gap="small"):
    _md('<span class="pil-brand pil-on-bg"><span class="dot"></span>Polymer Intelligence Lab</span>')
    with st.container(horizontal=True, key="pil-steps", gap="small", width="content"):
        for _page, _label in PAGES.items():
            st.button(
                _label,
                key=f"nav_{_page}",
                on_click=go_to,
                args=(_page,),
                type="primary" if st.session_state.page == _page else "secondary",
            )

page = st.session_state.page


def page_heading(eyebrow, title, description=None):
    desc = f'<p class="pil-page-desc">{description}</p>' if description else ""
    _md(
        '<div class="pil-page-head pil-on-bg">'
        f'<div class="pil-eyebrow">{eyebrow}</div>'
        f'<div class="pil-page-title" role="heading" aria-level="1">{title}</div>'
        f"{desc}</div>"
    )


def block_title(text):
    _md(f'<div class="pil-block-title pil-on-bg" role="heading" aria-level="2">{text}</div>')


def plot_release_curves(curves, title):
    """Draw one or more release curves. `curves` = [(t, C, label, color, dashed), ...]."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    for t_plot, C, label, color, dashed in curves:
        if dashed:
            ax.plot(t_plot, C, linewidth=2, color=color, linestyle="--", label=label)
        else:
            ax.plot(t_plot, C, linewidth=2.5, color=color, label=label)
            ax.fill_between(t_plot, C, alpha=0.12, color=color)
    style_release_axes(fig, ax, title, legend=True)
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# Page 1 -- Project overview
# ===========================================================================
if page == "overview":
    _md(
        '<div class="pil-hero pil-on-bg">'
        '<div class="pil-eyebrow">Structure → Property · Physics-informed ML</div>'
        "<h1>Polymer Intelligence Lab</h1>"
        '<p class="pil-hero-sub">AI-powered prediction of polymer properties from molecular structure.</p>'
        "</div>"
    )

    # Reflects which model files are actually present -- nothing is loaded here.
    status_bar([
        ("XGBoost · Tg", os.path.exists("models/xgboost_tg.joblib"), "Phase 1 model file"),
        ("GNN · Tg", GNN_AVAILABLE and os.path.exists("models/gnn_tg.pt"), "Phase 2 model file"),
        ("PINN · diffusion", True, "Phase 3: trained live on demand"),
        ("DeepONet · diffusion", deeponet_ready, "Phase 4 model file"),
    ])

    _md(
        '<div class="pil-intro pil-on-bg">A from-scratch machine learning project that predicts a polymer\'s '
        "<b>glass transition temperature (Tg)</b> from its repeat-unit structure, and models "
        "<b>drug release by diffusion</b> through a polymer slab with physics-informed learning. "
        "It progresses from classical ML through graph neural networks and physics-informed "
        "networks to a neural operator.</div>"
    )

    with st.container(key="pil-cta"):
        st.button(
            "Start",
            key="start_button",
            type="primary",
            icon=":material/arrow_forward:",
            on_click=go_to,
            args=("calculate",),
        )

    block_title("Objectives")
    _md(
        '<ol class="pil-objectives">'
        '<li><span class="n">01</span><span><b>Predict Tg from structure</b> &mdash; from a repeat-unit '
        "SMILES string alone, with no lab measurement.</span></li>"
        '<li><span class="n">02</span><span><b>Compare model families honestly</b> &mdash; fingerprints vs. '
        "molecular graphs, checked with scaffold splits.</span></li>"
        '<li><span class="n">03</span><span><b>Embed physics in learning</b> &mdash; solve the diffusion '
        "PDE (Fick's second law) with a physics-only loss.</span></li>"
        '<li><span class="n">04</span><span><b>Make simulation instant</b> &mdash; a neural operator that '
        "returns a release curve for any D in one forward pass.</span></li>"
        "</ol>"
    )

    block_title("Key features")
    _phase4_eval = load_deeponet_eval_results()
    if _phase4_eval is not None:
        _phase4_result = (
            f"MAE {_phase4_eval['mae']:.4f}, R² {_phase4_eval['r2']:.3f} "
            f"on held-out D values"
        )
    else:
        _phase4_result = "not yet validated (run deeponet_evaluate.py)"

    phase_grid([
        ("1", "Classical ML baseline", "Random Forest / XGBoost on Morgan fingerprints",
         "XGBoost MAE 27.5°C, R² 0.87"),
        ("2", "Graph Neural Network", "GINEConv message passing on the molecular graph, averaged with XGBoost",
         "Ensemble MAE 26.7°C, R² 0.855"),
        ("3", "Physics-Informed NN", "Diffusion PDE (Fick's second law), trained with a physics loss only",
         "Mean error 0.0062 vs. exact solution"),
        ("4", "Neural Operator", "DeepONet learning the diffusion PDE across many D values",
         _phase4_result),
    ])

    _md(
        '<div class="pil-feature-grid" style="margin-top:0.75rem">'
        '<div class="pil-feature"><div class="icon">🌡</div><div class="name">Tg prediction</div>'
        '<div class="desc">XGBoost and GNN predictions with their 50/50 ensemble and a glassy → rubber-like scale.</div></div>'
        '<div class="pil-feature"><div class="icon">💊</div><div class="name">Drug release curves</div>'
        '<div class="desc">Instant DeepONet curve, with an optional live-trained PINN overlay for the same D.</div></div>'
        '<div class="pil-feature"><div class="icon">📋</div><div class="name">Batch predictions</div>'
        '<div class="desc">Predict Tg for many SMILES at once and download the results as CSV.</div></div>'
        "</div>"
    )

    block_title("Project details")
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        with card("about-data"):
            card_header("Dataset")
            bullet_list([
                "<b>PolyMetriX curated Tg dataset</b>: 7,367 real polymers with repeat-unit "
                "SMILES, experimental Tg and source-reliability flags.",
                "Only the 4 <b>“red”</b> entries (conflicting values across sources) are dropped, "
                "leaving 7,363 polymers. “Black” means single-source, not wrong, so it is kept.",
            ])
        with card("about-rigor"):
            card_header("Rigor checks")
            bullet_list([
                "<b>Scaffold split</b> (no shared Murcko scaffold between train and test): "
                "XGBoost MAE 39.7°C, R² 0.772 — the more honest estimate for novel structures.",
                "<b>Ensemble weighting</b>: a search over XGBoost/GNN blend weights confirmed "
                "the 50/50 average as optimal.",
            ])
    with col_right:
        with card("about-stack"):
            card_header("Technology stack")
            tags(["RDKit", "scikit-learn", "XGBoost", "PyTorch", "PyTorch Geometric",
                     "pandas", "NumPy", "matplotlib", "Streamlit"])
        with card("about-limits"):
            card_header("Known limitations")
            bullet_list([
                "A few Tg predictions are badly off (150–300°C), mostly for structurally unusual "
                "polymers with few similar neighbors in the training data.",
                "The DeepONet is least accurate near the low-D extreme and can dip slightly "
                "below zero concentration at high D.",
                "D is not yet reliably predicted from structure: it is set manually, or comes "
                "from a permeability-derived proxy where that model is available.",
            ])

    with st.expander("Architecture diagram", icon=":material/account_tree:"):
        with st.container(key="pil-architecture"):
            st.image(os.path.join(os.path.dirname(__file__), "docs", "architecture.svg"), width="stretch")


# ===========================================================================
# Page 2 -- Calculations (all inputs + one Calculate button)
# ===========================================================================
elif page == "calculate":
    page_heading(
        "Step 2 of 3",
        "Calculations",
        "Choose a polymer, set the diffusion parameters and, optionally, a batch of SMILES. "
        "Calculate runs every model and opens the results.",
    )

    # ---- Polymer input + structure preview ----
    with card("selection", accent=True):
        col_input, col_structure = st.columns([1.8, 1], gap="large")

        with col_input:
            card_header(
                "Polymer input",
                "Choose a known polymer or paste a repeat-unit SMILES. Tg and drug release both use this structure.",
            )
            field_label("Polymer")
            dropdown_options = list(EXAMPLE_POLYMERS.keys()) + [CUSTOM_LABEL]
            st.selectbox(
                "Select polymer",
                dropdown_options,
                key="polymer_choice",
                on_change=_apply_dropdown_choice,
                label_visibility="collapsed",
            )

            field_label("Quick examples")
            with st.container(horizontal=True, gap="small", key="pil-chips"):
                for label, smiles in EXAMPLE_POLYMERS.items():
                    st.button(
                        label,
                        key=f"example_{label}",
                        on_click=_use_example_chip,
                        args=(label, smiles),
                        type="primary" if st.session_state.polymer_choice == label else "secondary",
                    )

            field_label("Repeat-unit SMILES")
            smiles_input = st.text_input(
                "Repeat-unit SMILES",
                key="smiles_value",
                help="The polymer's repeat-unit structure as a SMILES string.",
                label_visibility="collapsed",
                placeholder="e.g. CC(c1ccccc1)C",
            )

            _parsed_ok = bool(smiles_input) and Chem.MolFromSmiles(smiles_input) is not None
            if smiles_input and _parsed_ok:
                badge("✓ Valid SMILES — ready to predict", ok=True)
            elif smiles_input and not _parsed_ok:
                badge("⚠ Invalid SMILES — please enter a valid molecular structure", ok=False)
            else:
                hint("Select an example or type a SMILES string above.")

        with col_structure:
            with st.container(key="pil-structure"):
                field_label("Structure preview")
                img = render_molecule_image(smiles_input, size=(360, 260)) if _parsed_ok else None
                if img is not None:
                    st.image(img, width="stretch")
                    mol_caption(smiles_input)
                elif _parsed_ok:
                    empty_state("Structure preview unavailable for this input.", compact=True)
                else:
                    empty_state("Enter a valid SMILES string to preview its 2D structure.", compact=True)

    polymer_name = st.session_state.polymer_choice if st.session_state.polymer_choice != CUSTOM_LABEL else "this polymer"

    col_release, col_batch = st.columns(2, gap="medium")

    # ---- Drug release parameters (shared by the DeepONet and the PINN) ----
    with col_release:
        with card("release-params"):
            card_header(
                "Drug release parameters",
                "Effective diffusion coefficient for the release simulation.",
            )
            # Structure-driven D (Phase 3 permeability model), if available
            auto_D = get_structure_driven_D(smiles_input, _parsed_ok)
            if auto_D is not None:
                st.info(
                    f"🔬 Predicted CO₂ permeability suggests an effective diffusion "
                    f"coefficient of **D ≈ {auto_D:.2f}**. This is derived from the "
                    f"structure's predicted permeability as a physically-related "
                    f"proxy — not a literally measured drug-diffusion coefficient."
                )

            st.session_state.setdefault("use_structure_D", auto_D is not None)
            use_structure_D = st.checkbox(
                "Use structure-predicted D",
                disabled=(auto_D is None),
                key="use_structure_D",
            )

            field_label("Effective diffusion coefficient (D)")
            D_value = st.slider(
                "Effective diffusion coefficient (D)",
                min_value=0.01, max_value=1.0, step=0.01,
                help="Higher D = faster release. In the full project this comes from a permeability model; here it's a direct slider so you can explore.",
                label_visibility="collapsed",
                key="D_slider",
            )

            if use_structure_D and auto_D is not None:
                D_value = auto_D

            release_speed = "slow release" if D_value < 0.33 else "moderate release" if D_value < 0.66 else "fast release"
            readout("D", f"{D_value:.2f}", release_speed)

            train_pinn_too = st.checkbox(
                "Also train the PINN live for this D (about 20–30 seconds)",
                disabled=not deeponet_ready,  # without the DeepONet, the PINN is the only release model
                key="train_pinn",
                help="The DeepONet gives the curve instantly. The PINN trains from scratch for this exact D, "
                     "as a slower comparison. Curves are cached per D value.",
            )
            if not deeponet_ready:
                hint("The DeepONet model wasn't found (models/deeponet_diffusion.pt), so the release curve comes from the PINN.")

    # ---- Batch input ----
    with col_batch:
        with card("batch-input"):
            card_header("Batch Tg prediction", "Optional: one repeat-unit SMILES per line. Leave empty to skip.")
            batch_input = st.text_area(
                "SMILES (one per line)",
                height=150,
                label_visibility="collapsed",
                key="batch_text",
            )
            info_grid(
                [
                    ("Model", "XGBoost (Phase 1)"),
                    ("Blank lines", "Ignored"),
                    ("Invalid SMILES", "Kept, flagged"),
                    ("Export", "tg_predictions.csv"),
                ],
                mono_keys=("Export",),
            )

    # ---- Calculate ----
    with st.container(key="pil-cta"):
        calculate_clicked = st.button(
            "Calculate",
            key="calculate_button",
            disabled=not _parsed_ok,
            type="primary",
            icon=":material/calculate:",
        )
    if not _parsed_ok:
        _md('<p class="pil-hint pil-on-bg" style="text-align:center">Enter a valid SMILES string above to enable Calculate.</p>')

    if calculate_clicked and _parsed_ok:
        run = {
            "smiles": smiles_input,
            "name": polymer_name,
            "D": float(D_value),
            "D_from_structure": bool(use_structure_D and auto_D is not None),
            "tg": None, "tg_error": False,
            "deeponet": None, "pinn": None, "release_error": False,
            "batch": None,
        }

        # Tg: Phase 1 XGBoost, Phase 2 GNN when available, and their ensemble.
        with st.spinner("Analyzing molecular structure..."):
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
                run["tg"] = {"value": xgb_pred, "gnn_value": gnn_pred, "ensemble_value": ensemble_pred}
            except Exception:
                run["tg_error"] = True

        # Drug release: instant DeepONet curve, plus the live-trained PINN if requested.
        try:
            if deeponet_ready:
                run["deeponet"] = deeponet_release_curve(D_value)
            if train_pinn_too or not deeponet_ready:
                with st.spinner("Training physics-informed model... (about 20-30 seconds)"):
                    run["pinn"] = _cached_pinn_release_curve(round(D_value, 2))
        except Exception:
            run["release_error"] = True

        # Batch Tg (Phase 1 XGBoost), if any SMILES were given.
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        if lines:
            with st.spinner("Predicting Tg for each SMILES..."):
                model = load_tg_model()
                results = []
                for smi in lines:
                    X = featurize_smiles(smi)
                    if X is None:
                        results.append({"SMILES": smi, "Predicted Tg (°C)": None, "Status": "Invalid SMILES"})
                    else:
                        pred = model.predict(X)[0]
                        results.append({"SMILES": smi, "Predicted Tg (°C)": round(float(pred), 1), "Status": "OK"})
            run["batch"] = pd.DataFrame(results)

        st.session_state["last_run"] = run
        go_to("results")
        st.rerun()


# ===========================================================================
# Page 3 -- Results
# ===========================================================================
elif page == "results":
    run = st.session_state.get("last_run")
    page_heading(
        "Step 3 of 3",
        "Results",
        "Predicted glass transition temperature, the drug release curve and any batch predictions "
        "for the inputs you calculated.",
    )

    if run is None:
        empty_state("No results yet. Set up a polymer on the Calculations page and press Calculate.")
        with st.container(key="pil-cta"):
            st.button("Go to Calculations", key="to_calc_empty", type="primary",
                      icon=":material/arrow_back:", on_click=go_to, args=("calculate",))
    else:
        D_note = "structure-predicted" if run["D_from_structure"] else "manual"
        _md(
            '<div class="pil-run">'
            f'<span>Polymer <span class="v">{esc(run["name"])}</span></span>'
            f'<span>SMILES <span class="v mono">{esc(run["smiles"])}</span></span>'
            f'<span>D <span class="v mono">{run["D"]:.2f}</span> ({D_note})</span>'
            "</div>"
        )

        # ---- Glass transition temperature ----
        block_title("Glass transition temperature")
        if run["tg_error"] or run["tg"] is None:
            st.error("⚠ Something went wrong generating this prediction. Please try a different SMILES string.")
        else:
            tg = run["tg"]
            pred = tg["value"]
            # Map Tg onto a fixed, clearly-labeled visual scale. Purely a display
            # aid -- the underlying predicted value is used as-is everywhere else.
            scale_min, scale_max = -150.0, 250.0
            pct = max(0.0, min(1.0, (pred - scale_min) / (scale_max - scale_min))) * 100

            col_result, col_models, col_mol = st.columns([1.35, 1, 0.9], gap="medium")
            with col_result:
                result_card(
                    "Predicted glass transition temperature",
                    f"{pred:.1f}",
                    "°C",
                    f"{esc(run['name'])} · XGBoost (Phase 1)",
                    scale=(pct, scale_min, scale_max),
                )
            with col_models:
                # Multi-model comparison (Phase 1 XGBoost, Phase 2 GNN, ensemble)
                if tg.get("gnn_value") is not None:
                    st.metric("XGBoost (Phase 1)", f"{tg['value']:.1f} °C")
                    st.metric("GNN (Phase 2)", f"{tg['gnn_value']:.1f} °C")
                    st.metric("Ensemble average", f"{tg['ensemble_value']:.1f} °C")
                else:
                    st.metric("XGBoost (Phase 1)", f"{tg['value']:.1f} °C")
                    st.info("ℹ️ The Phase 2 GNN model wasn't found, so only the XGBoost prediction is shown.")
            with col_mol:
                with st.container(key="pil-structure"):
                    img = render_molecule_image(run["smiles"], size=(360, 260))
                    if img is not None:
                        st.image(img, width="stretch")
                        mol_caption(run["smiles"])

            callout(
                "<b>What this means:</b> below the glass transition temperature, "
                "the polymer behaves more like a rigid glass. Above it, the "
                "polymer becomes softer and more rubber-like. Prediction from an "
                "XGBoost model trained on Morgan fingerprints (Phase 1)."
            )

        # ---- Drug release ----
        block_title("Drug release")
        if run["release_error"]:
            st.error("⚠ The simulation could not complete. Please try again or adjust the diffusion coefficient.")
        elif run["deeponet"] is None and run["pinn"] is None:
            empty_state("No release curve was calculated.", compact=True)
        else:
            col_curve, col_info = st.columns([1.65, 1], gap="medium")
            with col_curve:
                with card("release-curve", accent=True):
                    card_header("Release curve", "Concentration at the slab center over time")
                    curves = []
                    if run["deeponet"] is not None:
                        t_plot, C_deeponet = run["deeponet"]
                        curves.append((t_plot, C_deeponet, "DeepONet (instant)", CURVE_PRIMARY, False))
                    if run["pinn"] is not None:
                        t_plot, C_pinn = run["pinn"]
                        # Dashed comparison when the DeepONet curve is also shown
                        curves.append((t_plot, C_pinn, "PINN (live-trained)",
                                       CURVE_SECONDARY if curves else CURVE_PRIMARY, bool(curves)))
                    plot_release_curves(curves, f"Predicted release curve: {run['name']} (D={run['D']:.2f})")
            with col_info:
                with card("release-info"):
                    card_header("How to read this")
                    if run["deeponet"] is not None and run["pinn"] is not None:
                        how = ("The solid cyan curve is the DeepONet's instant prediction; the dashed "
                               "violet curve is the PINN trained from scratch for this same D. The two "
                               "should look similar, illustrating the speed/accuracy tradeoff.")
                    else:
                        how = ("The curve shows how much drug concentration remains at the center of the "
                               "polymer slab over time. A steep early drop means fast initial release; "
                               "a curve that flattens out means release has mostly finished.")
                    _md(f'<p class="pil-hint" style="line-height:1.6">{how}</p>')
                    eval_results = load_deeponet_eval_results() if run["deeponet"] is not None else None
                    if eval_results is not None:
                        info_grid(
                            [
                                ("Held-out MAE", f"{eval_results['mae']:.4f}"),
                                ("Held-out R²", f"{eval_results['r2']:.3f}"),
                                ("Unseen D values", str(len(eval_results["D_values_tested"]))),
                            ],
                            mono_keys=("Held-out MAE", "Held-out R²", "Unseen D values"),
                        )
                    if run["pinn"] is not None:
                        info_grid(
                            [("Adam epochs", "800"), ("L-BFGS rounds", "5"), ("Probe point", "x = 0.5"), ("Time span", "0 – 5")],
                            mono_keys=("Adam epochs", "L-BFGS rounds", "Probe point", "Time span"),
                        )
                        hint("The PINN uses a reduced, faster training run than the fully validated "
                             "version: treat its curve's shape as illustrative, not precise.")
            st.caption(
                "The drug-release D is a permeability-derived proxy, not a literal "
                "drug-diffusion coefficient. See the GitHub README for full methodology and limitations."
            )

        # ---- Batch ----
        if run["batch"] is not None:
            block_title("Batch Tg predictions")
            results_df = run["batch"]
            n_valid = (results_df["Status"] == "OK").sum()
            n_invalid = len(results_df) - n_valid

            summary_cols = st.columns(3)
            summary_cols[0].metric("SMILES submitted", len(results_df))
            summary_cols[1].metric("Predicted", int(n_valid))
            summary_cols[2].metric("Invalid", int(n_invalid))

            st.dataframe(results_df, width="stretch", hide_index=True)

            if n_invalid > 0:
                st.warning(f"{n_invalid} SMILES could not be parsed and were skipped.")
            st.success(f"Predicted Tg for {n_valid} polymers.")

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "tg_predictions.csv", "text/csv",
                               icon=":material/download:")

        with st.container(key="pil-cta"):
            st.button("Edit inputs and recalculate", key="to_calc", type="secondary",
                      icon=":material/arrow_back:", on_click=go_to, args=("calculate",))

footer(
    "Built as a learning project: Classical ML → Graph Neural Networks → "
    "Physics-Informed Neural Networks → Neural Operators.<br>"
    "See the README for methodology and limitations."
)
