"""
Το App Control των Windows μπορεί να μπλοκάρει τα DLL του scikit-learn (ImportError στο `import sklearn`).
Ο κώδικας του project δεν πρέπει να το χρειάζεται, ώστε το training και τα tests να τρέχουν και σε τέτοια μηχανήματα.
"""
import ast
import os

import numpy as np
import pytest

from ml.training.train import train_test_split
from tests.conftest import ROOT

SCAN_DIRS = ("ml", "services", "scripts", "tests")


def test_project_code_does_not_import_sklearn():
    offenders = []
    for d in SCAN_DIRS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    mods = []
                    if isinstance(node, ast.Import):
                        mods = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        mods = [node.module]
                    if any(m == "sklearn" or m.startswith("sklearn.") for m in mods):
                        offenders.append(os.path.relpath(path, ROOT))
    assert not offenders, f"Εισαγωγή sklearn σε: {sorted(set(offenders))}"


def test_train_test_split_sizes_disjoint_and_deterministic():
    X = np.arange(50 * 3).reshape(50, 3)
    y = np.arange(50)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(Xva) == 10 and len(Xtr) == 40 and len(ytr) == 40 and len(yva) == 10
    assert set(ytr).isdisjoint(yva) and set(ytr) | set(yva) == set(y)  # κανένα δείγμα σε δύο σύνολα
    assert np.array_equal(Xtr[:, 0] // 3, ytr) and np.array_equal(Xva[:, 0] // 3, yva)  # X και y μένουν ευθυγραμμισμένα

    again = train_test_split(X, y, test_size=0.2, random_state=42)
    assert all(np.array_equal(a, b) for a, b in zip((Xtr, Xva, ytr, yva), again))
    other = train_test_split(X, y, test_size=0.2, random_state=7)
    assert not np.array_equal(other[3], yva)  # άλλο seed => άλλος διαχωρισμός


def test_train_test_split_rounds_val_up_like_sklearn():
    X, y = np.zeros((7, 1)), np.arange(7)
    _, Xva, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(Xva) == 2  # ceil(0.2 * 7)


@pytest.mark.parametrize("n", [0, 1])
def test_train_test_split_needs_at_least_two_samples(n):
    with pytest.raises(ValueError):
        train_test_split(np.zeros((n, 1)), np.zeros(n))


def test_train_test_split_rejects_mismatched_lengths_and_bad_fraction():
    with pytest.raises(ValueError):
        train_test_split(np.zeros((5, 1)), np.zeros(4))
    with pytest.raises(ValueError):
        train_test_split(np.zeros((5, 1)), np.zeros(5), test_size=1.5)
