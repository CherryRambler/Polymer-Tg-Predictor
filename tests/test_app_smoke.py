"""
test_app_smoke.py

Purpose: Catch the two ways the deployed app has broken before:
  1. A file app.py loads at runtime is missing from the repo (an earlier
     .gitignore change silently dropped the model files, and Streamlit
     Cloud crashed with FileNotFoundError).
  2. app.py itself raises an exception while rendering a page, or while
     running the models after "Calculate" is pressed.

Loading the app takes a few seconds and does NOT train the PINN (that
only happens after pressing Calculate with the PINN box ticked, which these
tests leave unticked), so the full AppTest run is cheap enough for CI.

Run: pytest tests/test_app_smoke.py -v
"""

import os
import re
import subprocess

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(ROOT, "app.py")

# Every file app.py reads at runtime. If you add one to app.py, add it here.
RUNTIME_FILES = [
    "models/xgboost_tg.joblib",
    "models/gnn_tg.pt",
    "models/deeponet_diffusion.pt",
    "results/deeponet_evaluation.json",
    "results/uncertainty.json",
    "data/raw/polymer_tg_polymetrix_reliable.csv",
    "docs/architecture.svg",
    "src/analysis/uncertainty.py",
    # Python modules app.py imports from src/
    "src/phase2_gnn/gnn_model.py",
    "src/phase2_gnn/gnn_inference.py",
    "src/phase3_pinn/pinn_model.py",
    "src/phase3_pinn/pinn_train.py",
    "src/phase4_neural_operator/deeponet_model.py",
]

MAX_FILE_MB = 50  # keep well under GitHub's 100 MB hard limit


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.splitlines())


def paths_referenced_in_app():
    """Find data paths written in app.py, either as 'models/x.pt' or as
    os.path.join(..., "results", "x.json")."""
    with open(APP_PATH, encoding="utf-8") as f:
        source = f.read()
    folders = "models|results|data|docs"
    direct = re.findall(rf"""["']((?:{folders})/[\w./-]+)["']""", source)
    joined = re.findall(rf"""["']({folders})["']\s*,\s*["']([\w.-]+)["']""", source)
    return set(direct) | {f"{folder}/{name}" for folder, name in joined}


@pytest.mark.parametrize("path", RUNTIME_FILES)
def test_runtime_file_exists_and_is_tracked(path):
    full = os.path.join(ROOT, path)
    assert os.path.exists(full), f"{path} is missing on disk"
    assert path in tracked_files(), f"{path} exists locally but is not committed (check .gitignore)"
    assert os.path.getsize(full) < MAX_FILE_MB * 1024 * 1024, f"{path} is over {MAX_FILE_MB} MB"


def test_every_data_path_in_app_is_listed():
    """Any models/, results/, data/ or docs/ path in app.py must be in
    RUNTIME_FILES, so this test can't silently go stale."""
    missing = paths_referenced_in_app() - set(RUNTIME_FILES)
    assert not missing, f"app.py loads {sorted(missing)} -- add them to RUNTIME_FILES"


@pytest.mark.parametrize("page", ["overview", "calculate", "results"])
def test_app_page_renders_without_exception(page, monkeypatch):
    monkeypatch.chdir(ROOT)  # app.py uses paths relative to the project root
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_calculate_runs_all_models_without_exception(monkeypatch):
    """Press Calculate with the default polymer (PINN box unticked), then
    check the Results page shows the three Tg numbers and no errors."""
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["page"] = "calculate"
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    at.button(key="calculate_button").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["page"] == "results"

    run = at.session_state["last_run"]
    assert not run["tg_error"] and not run["release_error"]
    assert run["tg"]["gnn_value"] is not None, "GNN prediction missing (model file not loaded?)"
    assert run["deeponet"] is not None, "DeepONet curve missing (model file not loaded?)"
    assert run["pinn"] is None, "the PINN should not train unless asked"
    labels = [m.label for m in at.metric]
    assert "Ensemble average" in labels
    assert run["tg"]["max_similarity"] is not None
    assert any("split-conformal" in c.value for c in at.caption), "uncertainty caption missing"
    assert not at.error, [e.value for e in at.error]
