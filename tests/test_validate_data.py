"""
test_validate_data.py

Purpose: Confirm our SMILES validation actually catches broken input.
This is the kind of check that would have caught our very first bug in
this project (the stray-space typo in the polyurethane SMILES that broke
the CSV) automatically, instead of us discovering it by reading terminal
output.

Run: pytest tests/test_validate_data.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "phase1_baseline"))

from validate_data import validate_smiles


def test_valid_smiles_accepted():
    assert validate_smiles("CC(c1ccccc1)C") is True  # polystyrene repeat unit


def test_simple_valid_smiles():
    assert validate_smiles("CC") is True  # polyethylene


def test_garbage_string_rejected():
    assert validate_smiles("not a molecule") is False


def test_malformed_smiles_rejected():
    assert validate_smiles("CC(C") is False


def test_empty_string_rejected():
    assert validate_smiles("") is False


def test_stray_space_in_middle_rejected():
    """
    This is the EXACT category of bug that broke our project early on:
    a stray space inside a SMILES string that silently corrupted a CSV
    row. This test exists specifically because that happened once.
    """
    assert validate_smiles("CC C") is False
