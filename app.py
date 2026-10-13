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
    # Surfaces (graphite)
    "bg": "#0C0E0F",
    "bg-2": "#111415",
    "surface": "#161A1B",
    "surface-2": "#1C2123",
    "border": "#2A3134",
    "border-strong": "#3A4447",
    # Accents
    "accent": "#2DD4A4",
    "accent-2": "#F59E0B",
    "accent-2-soft": "#FBBF24",
    # Text
    "text": "#E8ECEA",
    "text-muted": "#9AA5A1",
    "text-faint": "#66706C",
    # Status
    "success": "#4ADE80",
    "error": "#F87171",
    # Shape
    "radius-sm": "6px",
    "radius": "10px",
    "radius-lg": "12px",
    # Type
    "font-sans": "'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
    "font-mono": "'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace",
}

# Chart series colors -- shared by every matplotlib figure in the app.
CURVE_PRIMARY = TOKENS["accent"]
CURVE_SECONDARY = TOKENS["accent-2-soft"]

_CSS = """
/* ---------- Page canvas ---------- */
.stApp {
    background:
        radial-gradient(1000px 420px at 50% -12%, rgba(45, 212, 164, 0.06), transparent 65%),
        var(--pil-bg);
    font-family: var(--pil-font-sans);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding: 1.25rem 1.5rem 3rem;
}
#MainMenu, footer, header { visibility: hidden; }

/* ---------- Typography primitives ---------- */
.pil-mono { font-family: var(--pil-font-mono); }
.pil-eyebrow {
    font-family: var(--pil-font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pil-accent);
}
.pil-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--pil-text-muted);
    margin: 0.2rem 0 0.1rem 1px;
}

/* ---------- Hero ---------- */
.pil-hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 2.4rem 1.5rem 2rem;
    margin-bottom: 0.6rem;
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius-lg);
    background:
        linear-gradient(90deg, transparent, rgba(45, 212, 164, 0.8), rgba(245, 158, 11, 0.8), transparent) top / 100% 1px no-repeat,
        repeating-linear-gradient(0deg, rgba(154, 165, 161, 0.05) 0 1px, transparent 1px 32px),
        repeating-linear-gradient(90deg, rgba(154, 165, 161, 0.05) 0 1px, transparent 1px 32px),
        var(--pil-bg-2);
}
.pil-hero-art { display: flex; justify-content: center; margin-bottom: 0.6rem; }
.pil-hero-art svg { width: 56px; height: 56px; }
.pil-hero-title {
    font-family: var(--pil-font-sans);
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--pil-text);
    margin: 0.4rem 0 0.6rem;
    padding: 0;
}
.pil-hero-sub {
    font-size: 0.98rem;
    color: var(--pil-text-muted);
    max-width: 44ch;
    margin: 0 auto;
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
    background: rgba(22, 26, 27, 0.7);
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
    box-shadow: 0 0 8px rgba(74, 222, 128, 0.6);
}
.pil-status-item.ok .dot::after {
    content: "";
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    border: 1px solid rgba(74, 222, 128, 0.55);
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
        radial-gradient(circle at 30% 20%, rgba(45, 212, 164, 0.16), transparent 70%),
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

/* ---------- Cards (native bordered containers) ---------- */
[class*="st-key-pil-card-"], [class*="st-key-pil-accent-"] {
    background: linear-gradient(180deg, rgba(28, 33, 35, 0.92), rgba(22, 26, 27, 0.92));
    border-color: var(--pil-border) !important;
    border-radius: var(--pil-radius-lg) !important;
    padding: 1.2rem 1.3rem !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 14px 32px -18px rgba(0, 0, 0, 0.7);
    transition: border-color 0.2s ease;
}
[class*="st-key-pil-card-"]:hover { border-color: var(--pil-border-strong) !important; }
/* Emphasized card: thin emerald->amber edge along the top */
[class*="st-key-pil-accent-"] {
    background:
        linear-gradient(90deg, rgba(45, 212, 164, 0.7), rgba(245, 158, 11, 0.7)) top / 100% 1px no-repeat,
        radial-gradient(600px 160px at 0% 0%, rgba(45, 212, 164, 0.06), transparent 70%),
        linear-gradient(180deg, rgba(28, 33, 35, 0.95), rgba(22, 26, 27, 0.95));
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
    background: var(--pil-accent);
    box-shadow: 0 0 0 3px rgba(45, 212, 164, 0.15), 0 0 10px rgba(45, 212, 164, 0.5);
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
    background: linear-gradient(135deg, #2DD4A4 0%, #5EEAC0 100%);
    color: #04140E;
    border: 1px solid rgba(45, 212, 164, 0.9);
    font-weight: 600;
    min-height: 2.6rem;
    box-shadow: 0 6px 20px -8px rgba(45, 212, 164, 0.55);
    transition: box-shadow 0.18s ease, transform 0.18s ease, filter 0.18s ease;
}
[data-testid="stBaseButton-primary"]:hover {
    color: #04140E;
    border-color: #5EEAC0;
    filter: brightness(1.06);
    box-shadow: 0 8px 26px -8px rgba(45, 212, 164, 0.75);
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
    border-color: rgba(45, 212, 164, 0.6);
    color: #99F6D4;
    background: #20282A;
}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {
    outline: 2px solid var(--pil-accent);
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
    background: rgba(45, 212, 164, 0.12);
    color: #99F6D4;
    border-color: rgba(45, 212, 164, 0.65);
    box-shadow: 0 0 0 3px rgba(45, 212, 164, 0.08);
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
    box-shadow: 0 0 0 3px rgba(45, 212, 164, 0.16);
}
/* SMILES are chemistry, not prose: monospace wherever they are typed */
.st-key-smiles_value input, .st-key-pil-card-batch-input textarea {
    font-family: var(--pil-font-mono);
    font-size: 0.88rem;
}

/* ---------- Tabs: segmented navigation (ARIA roles: stable across versions) ---------- */
.stTabs [role="tablist"] {
    gap: 0.25rem;
    background: rgba(17, 20, 21, 0.85);
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
.stTabs [role="tab"]:hover { color: var(--pil-text); background: rgba(28, 33, 35, 0.8); }
.stTabs [role="tab"][aria-selected="true"] {
    background: var(--pil-surface-2);
    color: #99F6D4;
    box-shadow: inset 0 0 0 1px var(--pil-border-strong), inset 0 -2px 0 var(--pil-accent);
}
.stTabs [role="tab"][aria-selected="true"] p { color: #99F6D4; }
.stTabs [role="tab"]:focus-visible { outline: 2px solid var(--pil-accent); outline-offset: 1px; }
.stTabs [role="tabpanel"] { padding-top: 1rem; }

/* ---------- Metrics, expanders, images ---------- */
[data-testid="stMetric"] {
    background: rgba(12, 14, 15, 0.55);
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
    background: rgba(22, 26, 27, 0.6);
}
.st-key-pil-architecture [data-testid="stImage"] img { border-radius: 10px; }
/* RDKit draws black-on-white PNGs. Display-only dark rendering: invert +
   hue-rotate keeps heteroatom colors (O stays red, N blue), and the screen
   blend lets the card surface show through the (now black) background. */
.st-key-pil-structure [data-testid="stImage"] {
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius);
    background:
        radial-gradient(circle at 50% 40%, rgba(45, 212, 164, 0.07), transparent 70%),
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
.pil-badge.ok { color: var(--pil-success); background: rgba(74, 222, 128, 0.09); border-color: rgba(74, 222, 128, 0.28); }
.pil-badge.err { color: var(--pil-error); background: rgba(248, 113, 113, 0.09); border-color: rgba(248, 113, 113, 0.3); }
.pil-hint { font-size: 0.82rem; color: var(--pil-text-muted); margin: 0.1rem 0 0; }

/* ---------- Primary result ---------- */
.pil-result {
    position: relative;
    border-radius: var(--pil-radius-lg);
    border: 1px solid rgba(45, 212, 164, 0.35);
    background:
        radial-gradient(420px 200px at 15% 0%, rgba(45, 212, 164, 0.14), transparent 70%),
        radial-gradient(380px 200px at 100% 100%, rgba(245, 158, 11, 0.14), transparent 70%),
        var(--pil-bg-2);
    box-shadow: 0 0 0 1px rgba(45, 212, 164, 0.05), 0 18px 40px -20px rgba(45, 212, 164, 0.35);
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
.pil-result-value .unit { font-size: 1.35rem; color: var(--pil-accent); margin-left: 0.3rem; letter-spacing: 0; }
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
    background: linear-gradient(90deg, #2DD4A4 0%, #F59E0B 55%, #F87171 100%);
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
    box-shadow: 0 0 0 1px rgba(232, 236, 234, 0.5), 0 0 12px rgba(232, 236, 234, 0.45);
    transform: translate(-50%, -50%);
}

/* ---------- Callouts, readouts, empty states ---------- */
.pil-callout {
    font-size: 0.86rem;
    color: var(--pil-text-muted);
    line-height: 1.6;
    background: linear-gradient(90deg, rgba(45, 212, 164, 0.07), rgba(245, 158, 11, 0.03));
    border: 1px solid var(--pil-border);
    border-left: 2px solid var(--pil-accent);
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
    color: #FDE68A;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
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
    background: rgba(12, 14, 15, 0.4);
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
.pil-pipeline .step:last-child { border-color: rgba(45, 212, 164, 0.5); color: #99F6D4; }
.pil-pipeline .link {
    width: 22px;
    height: 1px;
    background: linear-gradient(90deg, rgba(45, 212, 164, 0.7), rgba(245, 158, 11, 0.7));
}
.pil-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.55rem;
}
.pil-grid .item {
    background: rgba(12, 14, 15, 0.5);
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
    background: linear-gradient(180deg, rgba(28, 33, 35, 0.92), rgba(22, 26, 27, 0.92));
    border: 1px solid var(--pil-border);
    border-radius: var(--pil-radius-lg);
    padding: 1rem 1.1rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.pil-phase:hover { border-color: rgba(45, 212, 164, 0.45); transform: translateY(-1px); }
.pil-phase .num {
    display: inline-grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    font-family: var(--pil-font-mono);
    font-size: 0.78rem;
    color: #99F6D4;
    background: rgba(45, 212, 164, 0.1);
    border: 1px solid rgba(45, 212, 164, 0.35);
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
.pil-list li::marker { color: var(--pil-accent); }
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
    .pil-hero-title { font-size: 1.8rem; }
    .pil-hero { padding: 1.6rem 1rem 1.4rem; }
    [class*="st-key-pil-card-"], [class*="st-key-pil-accent-"] { padding: 1rem !important; }
    .stTabs [role="tab"] { padding: 0.45rem 0.7rem; }
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
        '<stop offset="0" stop-color="#2DD4A4"/><stop offset="1" stop-color="#F59E0B"/>'
        "</linearGradient></defs>"
        f'<path d="M5 27 L14 13 L24 27 L34 13" fill="none" stroke="url(#{gid})" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="5" cy="27" r="2.6" fill="#2DD4A4"/>'
        '<circle cx="14" cy="13" r="2.6" fill="#7FC98A"/>'
        '<circle cx="24" cy="27" r="2.6" fill="#C8B04C"/>'
        '<circle cx="34" cy="13" r="2.6" fill="#F59E0B"/>'
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
    """About tab: one card per project phase. `phases` = [(num, name, approach, result), ...]."""
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
# Hero -- static banner at the top of the page: eyebrow, title, subtitle and
# a polymer-chain motif. Plain markup styled by _CSS (no iframe, no JS).
# ---------------------------------------------------------------------------
def hero(title, eyebrow, subtitle=None):
    sub = f'<p class="pil-hero-sub">{subtitle}</p>' if subtitle else ""
    _md(
        '<div class="pil-hero">'
        f'<div class="pil-hero-art">{_glyph_svg()}</div>'
        f'<div class="pil-eyebrow">{eyebrow}</div>'
        f'<h1 class="pil-hero-title">{title}</h1>'
        f"{sub}</div>"
    )


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

    Shared by the Drug Release (PINN) tab and the Instant Prediction
    (DeepONet) tab so they always agree on the same structure-to-D value
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
# Hero + model availability
# ---------------------------------------------------------------------------
HERO_SUBTITLE = "AI-powered prediction of polymer properties from molecular structure."

hero(
    title="Polymer Intelligence Lab",
    eyebrow="Structure → Property",
    subtitle=HERO_SUBTITLE,
)

# Reflects which model files are actually present -- nothing is loaded here.
status_bar([
    ("XGBoost · Tg", os.path.exists("models/xgboost_tg.joblib"), "Phase 1 model file"),
    ("GNN · Tg", GNN_AVAILABLE and os.path.exists("models/gnn_tg.pt"), "Phase 2 model file"),
    ("PINN · diffusion", True, "Phase 3: trained live in the Drug Release tab"),
    ("DeepONet · diffusion", DEEPONET_AVAILABLE and os.path.exists("models/deeponet_diffusion.pt"),
     "Phase 4 model file"),
])

# ---------------------------------------------------------------------------
# Prediction workspace -- polymer selection (shared by every tab)
# ---------------------------------------------------------------------------
with card("selection", accent=True):
    col_input, col_structure = st.columns([1.8, 1], gap="large")

    with col_input:
        card_header(
            "Polymer input",
            "Choose a known polymer or paste a repeat-unit SMILES. Every tab below uses this structure.",
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

# ---------------------------------------------------------------------------
# Feature tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab4, tab3, tab5 = st.tabs([
    ":material/device_thermostat: Glass Transition Temperature",
    ":material/medication: Drug Release",
    ":material/bolt: Instant Prediction",
    ":material/menu_book: About",
    ":material/table_view: Batch",
])

with tab1:
    section_header(
        "Phase 1 · Phase 2 — property prediction",
        "Glass transition temperature",
        "Predict Tg from the repeat-unit structure with XGBoost on Morgan fingerprints, "
        "cross-checked by the graph neural network when its weights are available.",
    )

    with card("tg-predict", accent=True):
        predict_clicked = st.button(
            "Predict glass transition temperature",
            key="tg_button",
            disabled=not _parsed_ok,
            width="stretch",
            type="primary",
            icon=":material/online_prediction:",
        )
        if not _parsed_ok and not predict_clicked:
            hint("Enter a valid SMILES string above to enable this button.")

        if predict_clicked:
            if not _parsed_ok:
                badge("⚠ Invalid SMILES — please enter a valid molecular structure before predicting", ok=False)
            else:
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

            col_result, col_models = st.columns([1.35, 1], gap="medium")
            with col_result:
                result_card(
                    "Predicted glass transition temperature",
                    f"{pred:.1f}",
                    "°C",
                    f"{esc(result['name'])} · XGBoost (Phase 1)",
                    scale=(pct, scale_min, scale_max),
                )

            # ---- Multi-model comparison (Phase 1 XGBoost, Phase 2 GNN, ensemble) ----
            with col_models:
                field_label("Model comparison")
                if result.get("gnn_value") is not None:
                    st.metric("XGBoost (Phase 1)", f"{result['value']:.1f} °C")
                    st.metric("GNN (Phase 2)", f"{result['gnn_value']:.1f} °C")
                    st.metric("Ensemble average", f"{result['ensemble_value']:.1f} °C")
                else:
                    st.info("ℹ️ The Phase 2 GNN model wasn't found, so only the XGBoost prediction is shown.")

            callout(
                "<b>What this means:</b> below the glass transition temperature, "
                "the polymer behaves more like a rigid glass. Above it, the "
                "polymer becomes softer and more rubber-like. Prediction from an "
                "XGBoost model trained on Morgan fingerprints (Phase 1)."
            )
        elif not predict_clicked:
            empty_state("Run a prediction to see the predicted Tg, the model comparison and where it sits on the glassy → rubber-like scale.")

    # ---- Model information (compact) ----
    with card("tg-model"):
        card_header("Model information")
        pipeline(["SMILES", "Molecular Fingerprint", "XGBoost", "Tg Prediction"])
        info_grid(
            [
                ("Model", "XGBoost"),
                ("Input", "Polymer SMILES"),
                ("Features", "Morgan Fingerprints"),
                ("Output", "Glass Transition Temp."),
                ("Unit", "°C"),
                ("Stage", "Phase 1"),
            ]
        )

with tab2:
    section_header(
        "Phase 3 — physics-informed neural network",
        "Drug release simulation",
        "Train a PINN on Fick's second law for the chosen effective diffusion coefficient D, "
        "then plot how concentration at the center of the polymer slab falls over time.",
    )
    callout(
        "This trains a small physics-informed neural network live, using a "
        "diffusion coefficient placeholder. This demo uses a <b>reduced, "
        "faster</b> training run than our fully validated version &mdash; "
        "treat the curve's overall <b>shape</b> as illustrative, not precise. "
        "Training takes about 20&ndash;30 seconds."
    )

    # ---- Structure-driven D (Phase 3 permeability model), if available ----
    auto_D = get_structure_driven_D(smiles_input, _parsed_ok)

    col_params, col_curve = st.columns([1, 1.65], gap="medium")
    with col_params:
        with card("pinn-params"):
            card_header("Simulation parameters")
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

            field_label("Effective diffusion coefficient (D)")
            D_manual = st.slider(
                "Effective diffusion coefficient (D)",
                min_value=0.01, max_value=1.0, value=0.3, step=0.01,
                help="Higher D = faster release. In the full project this comes from a permeability model; here it's a direct slider so you can explore.",
                label_visibility="collapsed",
            )

            if use_structure_D and auto_D is not None:
                D_manual = auto_D

            release_speed = "slow release" if D_manual < 0.33 else "moderate release" if D_manual < 0.66 else "fast release"
            readout("D", f"{D_manual:.2f}", release_speed)

            run_clicked = st.button(
                "Train PINN and show release curve",
                key="pinn_button",
                disabled=not _parsed_ok,
                width="stretch",
                type="primary",
                icon=":material/model_training:",
            )
            if not _parsed_ok and not run_clicked:
                hint("Enter a valid SMILES string above to enable this button.")

            info_grid(
                [("Adam epochs", "800"), ("L-BFGS rounds", "5"), ("Probe point", "x = 0.5"), ("Time span", "0 – 5")],
                mono_keys=("Adam epochs", "L-BFGS rounds", "Probe point", "Time span"),
            )

    with col_curve:
        with card("pinn-curve", accent=True):
            card_header("Release curve", "Concentration at the slab center over time")
            if run_clicked:
                try:
                    D_key = round(D_manual, 2)
                    with st.spinner("Training physics-informed model... (about 20-30 seconds)"):
                        t_plot, C_center = _cached_pinn_release_curve(D_key)
                    st.caption(f"Trained with D = {D_key:.2f} (this exact value determines the curve shape).")

                    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
                    ax.plot(t_plot, C_center, linewidth=2.5, color=CURVE_PRIMARY, label="PINN (live-trained)")
                    ax.fill_between(t_plot, C_center, alpha=0.12, color=CURVE_PRIMARY)
                    style_release_axes(fig, ax, f"Predicted release curve: {polymer_name} (D={D_manual:.2f})", legend=True)
                    st.pyplot(fig)
                    plt.close(fig)
                    callout(
                        "<b>How to read this:</b> the curve shows how much drug "
                        "concentration remains at the center of the polymer slab over "
                        "time. A steep early drop means fast initial release; a curve "
                        "that flattens out means release has mostly finished."
                    )
                except Exception:
                    st.error("⚠ The simulation could not complete. Please try again or adjust the diffusion coefficient.")
            else:
                empty_state(
                    "Set D and train the PINN to plot the release curve. "
                    "Curves are cached per D value, so re-running the same D is instant."
                )

with tab4:
    deeponet_ready = DEEPONET_AVAILABLE and os.path.exists("models/deeponet_diffusion.pt")
    eval_results = load_deeponet_eval_results() if deeponet_ready else None

    if eval_results is not None:
        accuracy_note = (
            f"validated MAE {eval_results['mae']:.4f}, "
            f"R² {eval_results['r2']:.3f} on {len(eval_results['D_values_tested'])} "
            f"D values never seen during training"
        )
    else:
        accuracy_note = "accuracy not yet validated -- run deeponet_evaluate.py"

    section_header(
        "Phase 4 — neural operator",
        "Instant release prediction",
        "A DeepONet trained once, offline, across many D values: each new D is a single forward pass, "
        "so the release curve updates as you move the slider.",
    )
    callout(
        f"<b>What's different here:</b> the Drug Release tab trains a fresh "
        f"PINN from scratch every time you change D (~20&ndash;30 seconds). "
        f"This tab instead uses a <b>DeepONet</b> (a neural operator) "
        f"trained once, offline, across many D values &mdash; evaluating a "
        f"new D afterward is a single forward pass, so the curve updates "
        f"instantly as you move the slider. The tradeoff is a small drop in "
        f"accuracy versus training a dedicated PINN for that exact D "
        f"({accuracy_note})."
    )

    if not deeponet_ready:
        st.info(
            "ℹ️ The trained DeepONet model wasn't found "
            "(models/deeponet_diffusion.pt). Run "
            "`python src/phase4_neural_operator/train_deeponet.py` to "
            "enable this tab."
        )
    else:
        auto_D_instant = get_structure_driven_D(smiles_input, _parsed_ok)

        col_controls, col_instant = st.columns([1, 1.65], gap="medium")
        with col_controls:
            with card("instant-controls"):
                card_header("Operator input")
                if auto_D_instant is not None:
                    st.info(
                        f"🔬 Predicted CO₂ permeability suggests an effective "
                        f"diffusion coefficient of **D ≈ {auto_D_instant:.2f}** "
                        f"(same structure-to-D pipeline as the Drug Release tab)."
                    )

                use_structure_D_instant = st.checkbox(
                    "Use structure-predicted D",
                    value=(auto_D_instant is not None),
                    disabled=(auto_D_instant is None),
                    key="instant_use_structure_D",
                )

                field_label("Effective diffusion coefficient (D)")
                D_instant = st.slider(
                    "Effective diffusion coefficient (D)",
                    min_value=0.01, max_value=1.0, value=0.3, step=0.01,
                    help="Move this slider -- the curve below updates instantly, no retraining.",
                    label_visibility="collapsed",
                    key="instant_D_slider",
                )

                if use_structure_D_instant and auto_D_instant is not None:
                    D_instant = auto_D_instant

                release_speed_instant = (
                    "slow release" if D_instant < 0.33
                    else "moderate release" if D_instant < 0.66
                    else "fast release"
                )
                readout("D", f"{D_instant:.2f}", release_speed_instant)

                compare_with_pinn = st.checkbox(
                    "Also show the PINN's live-trained curve for the same D (slower, for comparison)",
                    value=False,
                    disabled=not _parsed_ok,
                )

                if eval_results is not None:
                    info_grid(
                        [
                            ("Held-out MAE", f"{eval_results['mae']:.4f}"),
                            ("Held-out R²", f"{eval_results['r2']:.3f}"),
                            ("Unseen D values", str(len(eval_results["D_values_tested"]))),
                        ],
                        mono_keys=("Held-out MAE", "Held-out R²", "Unseen D values"),
                    )

        with col_instant:
            with card("instant-curve", accent=True):
                card_header("Release curve", "DeepONet forward pass at the slab center")
                if not _parsed_ok:
                    empty_state("Enter a valid SMILES string above to see a release curve.")
                else:
                    try:
                        t_plot, C_deeponet = deeponet_release_curve(D_instant)

                        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
                        ax.plot(t_plot, C_deeponet, linewidth=2.5, color=CURVE_PRIMARY,
                                label="DeepONet (instant)")
                        ax.fill_between(t_plot, C_deeponet, alpha=0.12, color=CURVE_PRIMARY)

                        if compare_with_pinn:
                            with st.spinner("Training PINN for comparison... (about 20-30 seconds)"):
                                _, C_pinn = _cached_pinn_release_curve(round(D_instant, 2))
                            ax.plot(t_plot, C_pinn, linewidth=2, color=CURVE_SECONDARY, linestyle="--",
                                    label="PINN (live-trained)")

                        style_release_axes(fig, ax, f"Instant release curve: {polymer_name} (D={D_instant:.2f})", legend=True)
                        st.pyplot(fig)
                        plt.close(fig)

                        callout(
                            "<b>How to read this:</b> the solid emerald curve is the "
                            "DeepONet's instant prediction. If you enabled the "
                            "comparison, the dashed amber curve is what the Phase 3 PINN "
                            "predicts after training from scratch for this same D "
                            "&mdash; the two should look similar, illustrating the "
                            "speed/accuracy tradeoff described above."
                        )
                    except Exception:
                        st.error("⚠ The instant prediction could not complete. Please try a different SMILES string or D value.")

with tab3:
    section_header(
        "Project overview",
        "About this project",
        "A from-scratch ML project predicting polymer properties from molecular structure, built in four phases.",
    )
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
                     "pandas", "NumPy", "matplotlib", "Streamlit", "React + Vite"])
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

    st.caption(
        "The drug-release D shown in the Drug Release tab is a "
        "permeability-derived proxy, not a literal drug-diffusion "
        "coefficient. See the GitHub README for full methodology and "
        "limitations."
    )

with tab5:
    section_header(
        "Phase 1 — XGBoost at scale",
        "Batch Tg prediction",
        "Paste multiple SMILES strings, one per line, to predict Tg for all of them at once.",
    )

    col_batch, col_format = st.columns([1.65, 1], gap="medium")
    with col_batch:
        with card("batch-input"):
            card_header("SMILES list", "One repeat-unit SMILES per line.")
            batch_input = st.text_area(
                "SMILES (one per line)",
                value="CC(c1ccccc1)C\nCC\nCC(C)(C(=O)OC)C",
                height=150,
                label_visibility="collapsed",
            )
            batch_clicked = st.button(
                "Predict all",
                key="batch_button",
                type="primary",
                width="stretch",
                icon=":material/playlist_play:",
            )
    with col_format:
        with card("batch-format"):
            card_header("Format")
            info_grid(
                [
                    ("Input", "Plain text, one SMILES per line"),
                    ("Blank lines", "Ignored"),
                    ("Model", "XGBoost (Phase 1)"),
                    ("Invalid SMILES", "Kept, flagged in Status"),
                    ("Export", "tg_predictions.csv"),
                ],
                mono_keys=("Export",),
            )

    if batch_clicked:
        with card("batch-results", accent=True):
            card_header("Results")
            with st.spinner("Predicting Tg for each SMILES..."):
                model = load_tg_model()
                lines = [line.strip() for line in batch_input.split("\n") if line.strip()]

                results = []
                for smi in lines:
                    X = featurize_smiles(smi)
                    if X is None:
                        results.append({"SMILES": smi, "Predicted Tg (°C)": None, "Status": "Invalid SMILES"})
                    else:
                        pred = model.predict(X)[0]
                        results.append({"SMILES": smi, "Predicted Tg (°C)": round(float(pred), 1), "Status": "OK"})

            results_df = pd.DataFrame(results)
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

footer(
    "Built as a learning project: Classical ML → Graph Neural Networks → "
    "Physics-Informed Neural Networks → Neural Operators.<br>"
    "See the README for methodology and limitations."
)
