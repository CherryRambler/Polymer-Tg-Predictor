"""
gen_architecture.py

Purpose: Generate docs/architecture.svg, the architecture diagram embedded
in the README. The layout is hand-placed on a fixed grid (NODES, EDGES,
BOXES below) so edge labels never overlap and edges never wrap around the
figure -- an auto-layout tool like Mermaid can't guarantee either.

To change the diagram, edit NODES / EDGES / BOXES and re-run. Coordinates
are in SVG pixels; every node in a column shares that column's x-range.

Run (from the project root):
    python docs/gen_architecture.py
"""
import os
import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "architecture.svg")
W, H = 1060, 1012

# Column x-ranges
COL = {"A": (32, 208), "B": (270, 446), "C": (500, 676), "a": (736, 876), "b": (920, 1036)}

GROUP = {
    "data": dict(stroke="#2563eb", fill="#eff6ff", text="#1e3a8a"),
    "tg":   dict(stroke="#16a34a", fill="#f0fdf4", text="#14532d"),
    "diff": dict(stroke="#ea580c", fill="#fff7ed", text="#7c2d12"),
    "eval": dict(stroke="#9333ea", fill="#faf5ff", text="#581c87"),
    "app":  dict(stroke="#db2777", fill="#fdf2f8", text="#831843"),
}

# id: (col, cy, lines, group, shape)   shape: script | file | input | output
NODES = {
    # 1 · Tg data pipeline
    "val":   ("A", 60,  ["validate_data.py", "SMILES parse check"], "data", "script"),
    "get":   ("A", 124, ["get_data.py", "convert PolyMetriX CSV"], "data", "script"),
    "filt":  ("A", 188, ["filter_reliable_data.py", "drop 'red' rows"], "data", "script"),
    "rel":   ("A", 280, ["polymer_tg_polymetrix", "_reliable.csv"], "data", "file2"),
    "feat":  ("A", 356, ["featurize.py", "Morgan fingerprints"], "data", "script"),
    "gdata": ("A", 476, ["gnn_data.py", "molecular graphs"], "data", "script"),
    # 4 · Tg evaluation
    "scaf":  ("B", 124, ["scaffold_split_eval.py", "Murcko scaffold CV"], "eval", "script"),
    "cmp":   ("C", 60,  ["compare_phase1_vs_gnn.py", "same 737 test polymers"], "eval", "script"),
    "ens":   ("C", 124, ["ensemble.py", "50/50 average"], "eval", "script"),
    "lens":  ("C", 188, ["learned_ensemble.py", "blend-weight search"], "eval", "script"),
    # 2 · Tg models
    "preds": ("C", 280, ["cv_predictions.csv", "gnn_test_predictions.csv"], "tg", "file2"),
    "tb":    ("B", 356, ["train_baseline.py", "XGBoost + RF, 5-fold CV"], "tg", "script"),
    "xgbw":  ("C", 356, ["xgboost_tg.joblib"], "tg", "file"),
    "gt":    ("B", 476, ["gnn_train.py", "80/10/10 split"], "tg", "script"),
    "gnnw":  ("C", 476, ["gnn_tg.pt"], "tg", "file"),
    "gm":    ("B", 542, ["gnn_model.py", "GINEConv architecture"], "tg", "script"),
    # 3 · Diffusion models
    "pm":    ("A", 648, ["pinn_model.py", "tanh MLP (x, t) → C"], "diff", "script"),
    "pt":    ("B", 648, ["pinn_train.py", "Adam + L-BFGS"], "diff", "script"),
    "pw":    ("B", 712, ["pinn_diffusion.pt", "offline run, D = 0.1"], "diff", "file"),
    "ana":   ("B", 776, ["pinn_analytical.py", "exact solution"], "diff", "script"),
    "gen":   ("B", 840, ["generate_training_data.py", "100 D values, 25×25 grid"], "diff", "script"),
    "dm":    ("A", 904, ["deeponet_model.py", "branch (D) + trunk (x, t)"], "diff", "script"),
    "td":    ("B", 904, ["train_deeponet.py", "Adam, 600 epochs"], "diff", "script"),
    "dw":    ("B", 968, ["deeponet_diffusion.pt", "DeepONet weights"], "diff", "file"),
    # 4 · Diffusion evaluation
    "pe":    ("C", 712, ["pinn_evaluate.py", "PINN vs exact, D = 0.1"], "eval", "script"),
    "de":    ("C", 840, ["deeponet_evaluate.py", "8 held-out D values"], "eval", "script"),
    # 5 · Streamlit app
    "xgbp":  ("a", 356, ["XGBoost Tg"], "app", "script"),
    "smi":   ("a", 416, ["SMILES input"], "app", "input"),
    "gnnp":  ("a", 476, ["GNN Tg", "gnn_inference.py"], "app", "script"),
    "ensn":  ("b", 416, ["Ensemble Tg", "(XGB + GNN) / 2"], "app", "output"),
    "pinnl": ("a", 648, ["PINN", "trained live"], "app", "script"),
    "dsl":   ("a", 808, ["D slider", "manual, 0.01–1.0"], "app", "input"),
    "don":   ("a", 968, ["DeepONet", "instant"], "app", "script"),
    "rc":    ("b", 808, ["Release curve"], "app", "output"),
}

# Solid line segments drawn without an arrowhead (the gap at x=458 marks a crossing).
NO_ARROW = [[(446, 356), (454, 356)]]

# (points, dashed, [labels as (x, y, text, anchor)])
EDGES = [
    # 1 · data pipeline
    ([(120, 104), (120, 80)], True, [(126, 96, "checked by", "start")]),
    ([(120, 144), (120, 168)], False, [(126, 160, "writes CSV for", "start")]),
    ([(120, 208), (120, 260)], False, [(126, 238, "writes", "start")]),
    ([(120, 300), (120, 336)], False, [(126, 330, "feeds", "start")]),
    ([(120, 312), (22, 312), (22, 476), (32, 476)], False, []),           # fork of "feeds"
    ([(208, 280), (240, 280), (240, 124), (270, 124)], False, [(264, 204, "re-splits", "start")]),
    ([(208, 356), (270, 356)], False, [(239, 350, "feeds", "middle")]),
    ([(208, 476), (270, 476)], False, [(239, 470, "feeds", "middle")]),
    # 2 · Tg models: script -> weights
    ([(358, 522), (358, 496)], False, [(364, 513, "defines", "start")]),
    # tb -> xgbw; its first stub is in NO_ARROW so the dashed gt -> preds line visibly crosses it
    ([(462, 356), (500, 356)], False, [(481, 350, "saves", "middle")]),
    ([(446, 476), (500, 476)], False, [(481, 490, "saves", "middle")]),
    ([(358, 336), (358, 272), (500, 272)], True, [(429, 266, "saved predictions", "middle")]),
    ([(446, 466), (458, 466), (458, 288), (500, 288)], True, [(452, 420, "saved predictions", "end")]),
    # 4 · Tg evaluation reads
    ([(588, 260), (588, 208)], True, [(594, 238, "read by", "start")]),
    ([(676, 280), (702, 280), (702, 124), (676, 124)], True, [(696, 96, "read by", "end")]),
    ([(702, 124), (702, 60), (676, 60)], True, []),
    # weights -> app
    ([(676, 356), (736, 356)], False, [(706, 350, "loads", "middle")]),
    ([(676, 476), (736, 476)], False, [(706, 470, "loads", "middle")]),
    # 5 · app, Tg
    ([(806, 401), (806, 376)], False, [(812, 392, "featurizes", "start")]),
    ([(806, 431), (806, 456)], False, [(812, 447, "graphs", "start")]),
    ([(876, 356), (978, 356), (978, 396)], False, [(927, 350, "averages", "middle")]),
    ([(876, 476), (978, 476), (978, 436)], False, [(927, 490, "averages", "middle")]),
    # 3 · diffusion models
    ([(208, 648), (270, 648)], False, [(239, 642, "defines", "middle")]),
    ([(358, 668), (358, 692)], False, [(364, 684, "saves", "start")]),
    ([(446, 648), (736, 648)], False, [(591, 642, "trains live", "middle")]),
    ([(358, 796), (358, 820)], False, [(364, 812, "labels", "start")]),
    ([(358, 860), (358, 884)], False, [(364, 876, "feeds", "start")]),
    ([(208, 904), (270, 904)], False, [(239, 898, "defines", "middle")]),
    ([(358, 924), (358, 948)], False, [(364, 940, "saves", "start")]),
    ([(446, 976), (736, 976)], False, [(640, 970, "loads", "middle")]),
    # 4 · diffusion evaluation
    ([(446, 712), (500, 712)], False, [(473, 706, "loads", "middle")]),
    ([(546, 732), (546, 768), (446, 768)], False, [(496, 762, "compares against", "middle")]),
    ([(446, 960), (560, 960), (560, 860)], False, [(566, 916, "loads", "start")]),
    ([(606, 820), (606, 784), (446, 784)], False, [(526, 798, "compares against", "middle")]),
    ([(676, 840), (706, 840), (706, 958), (736, 958)], True, [(700, 900, "reports MAE / R²", "end")]),
    # 5 · app, diffusion
    ([(806, 788), (806, 668)], False, [(812, 732, "sets D", "start")]),
    ([(806, 828), (806, 948)], False, [(812, 892, "sets D", "start")]),
    ([(876, 648), (978, 648), (978, 788)], False, [(927, 642, "plots", "middle")]),
    ([(876, 968), (978, 968), (978, 828)], False, [(927, 982, "plots", "middle")]),
]

BOXES = [  # (x1, y1, x2, y2, title, group)
    (12, 12, 222, 572, "1 · Tg data pipeline", "data"),
    (258, 12, 688, 218, "4 · Tg evaluation", "eval"),
    (258, 232, 688, 572, "2 · Tg models", "tg"),
    (12, 600, 458, 1000, "3 · Diffusion models", "diff"),
    (488, 664, 688, 870, "4 · Diffusion evaluation", "eval"),
    (720, 300, 1048, 1000, "5 · Streamlit app (app.py)", "app"),
]

JUNCTIONS = [(120, 312), (702, 124)]

out = []
w = out.append
w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">')
w('<title>Polymer-Tg-Predictor architecture</title>')
w('<style>'
  '.n1{font-size:11.5px;font-weight:600}'
  '.n2{font-size:10px;fill:#64748b}'
  '.gt{font-size:12.5px;font-weight:700}'
  '.el{font-size:10px;fill:#475569;stroke:#ffffff;stroke-width:4px;stroke-linejoin:round;paint-order:stroke}'
  '.e{fill:none;stroke:#64748b;stroke-width:1.4}'
  '.d{stroke-dasharray:5 4}'
  '.lg{font-size:10.5px;fill:#334155}'
  '</style>')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>')
w(f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff"/>')

for x1, y1, x2, y2, title, g in BOXES:
    c = GROUP[g]
    w(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="10" fill="{c["fill"]}" '
      f'fill-opacity="0.55" stroke="{c["stroke"]}" stroke-width="1.3"/>')
    w(f'<text class="gt" x="{x1+10}" y="{y1+17}" fill="{c["stroke"]}">{escape(title)}</text>')

def edge_path(pts):
    return "M" + " L".join(f"{x},{y}" for x, y in pts)

# Edges first, so nodes sit on top of line ends; labels last.
labels = []
for pts in NO_ARROW:
    w(f'<path class="e" d="{edge_path(pts)}"/>')
for pts, dashed, lbls in EDGES:
    cls = "e d" if dashed else "e"
    w(f'<path class="{cls}" d="{edge_path(pts)}" marker-end="url(#ah)"/>')
    labels.extend(lbls)

for x, y in JUNCTIONS:
    w(f'<circle cx="{x}" cy="{y}" r="2.6" fill="#64748b"/>')

def node(nid, col, cy, lines, g, shape):
    c = GROUP[g]
    x1, x2 = COL[col]
    h = 30 if shape == "input" and len(lines) == 1 else 40
    y1 = cy - h / 2
    wd = x2 - x1
    sw = 2.2 if shape == "output" else 1.2
    if shape in ("file", "file2"):
        f = 9  # folded corner
        w(f'<path d="M{x1},{y1} H{x2-f} L{x2},{y1+f} V{y1+h} H{x1} Z" fill="#ffffff" stroke="{c["stroke"]}" stroke-width="{sw}"/>')
        w(f'<path d="M{x2-f},{y1} V{y1+f} H{x2}" fill="none" stroke="{c["stroke"]}" stroke-width="1"/>')
    else:
        rx = h / 2 if shape == "input" else 6
        w(f'<rect x="{x1}" y="{y1}" width="{wd}" height="{h}" rx="{rx}" fill="#ffffff" '
          f'stroke="{c["stroke"]}" stroke-width="{sw}"/>')
    cx = (x1 + x2) / 2
    if len(lines) == 1:
        w(f'<text class="n1" x="{cx}" y="{cy+4}" text-anchor="middle" fill="{c["text"]}">{escape(lines[0])}</text>')
    else:
        second = "n1" if shape == "file2" else "n2"
        fill2 = f' fill="{c["text"]}"' if shape == "file2" else ""
        w(f'<text class="n1" x="{cx}" y="{cy-3}" text-anchor="middle" fill="{c["text"]}">{escape(lines[0])}</text>')
        w(f'<text class="{second}" x="{cx}" y="{cy+11}" text-anchor="middle"{fill2}>{escape(lines[1])}</text>')

for nid, spec in NODES.items():
    node(nid, *spec)

for x, y, t, anchor in labels:
    w(f'<text class="el" x="{x}" y="{y}" text-anchor="{anchor}">{escape(t)}</text>')

# Legend (top-right)
lx, ly = 740, 24
w(f'<rect x="{lx}" y="{ly}" width="300" height="252" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>')
w(f'<text class="gt" x="{lx+12}" y="{ly+20}" fill="#334155">Legend</text>')
row = ly + 44
def lg_text(y, t):
    w(f'<text class="lg" x="{lx+86}" y="{y+4}">{escape(t)}</text>')
w(f'<rect x="{lx+14}" y="{row-10}" width="58" height="20" rx="5" fill="#ffffff" stroke="#64748b"/>'); lg_text(row, "script / module")
row += 30
w(f'<path d="M{lx+14},{row-10} H{lx+64} L{lx+72},{row-2} V{row+10} H{lx+14} Z" fill="#ffffff" stroke="#64748b"/>'
  f'<path d="M{lx+64},{row-10} V{row-2} H{lx+72}" fill="none" stroke="#64748b"/>'); lg_text(row, "saved file (CSV, weights)")
row += 30
w(f'<rect x="{lx+14}" y="{row-10}" width="58" height="20" rx="10" fill="#ffffff" stroke="#64748b"/>'); lg_text(row, "user input in the app")
row += 30
w(f'<rect x="{lx+14}" y="{row-10}" width="58" height="20" rx="5" fill="#ffffff" stroke="#64748b" stroke-width="2.2"/>'); lg_text(row, "what the app shows")
row += 30
w(f'<path class="e" d="M{lx+14},{row} H{lx+72}" marker-end="url(#ah)"/>'); lg_text(row, "code / data flow")
row += 30
w(f'<path class="e d" d="M{lx+14},{row} H{lx+72}" marker-end="url(#ah)"/>'); lg_text(row, "offline read of a saved file, or a check")
row += 30
sw_x = lx + 14
for g, name in [("data", "data"), ("tg", "Tg"), ("diff", "diffusion"), ("eval", "eval"), ("app", "app")]:
    c = GROUP[g]
    w(f'<rect x="{sw_x}" y="{row-6}" width="10" height="10" rx="2" fill="{c["fill"]}" stroke="{c["stroke"]}"/>')
    w(f'<text class="lg" x="{sw_x+14}" y="{row+3}">{name}</text>')
    sw_x += 14 + len(name) * 6 + 14

w('</svg>')
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT)
