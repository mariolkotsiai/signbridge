"""scripts/inspect_sentences.py: διάγνωση των δεδομένων προτάσεων (ίδιος signer σε dev/test, παραλλαγές labels, χέρια)."""
import csv
import importlib.util
import json
import os

import numpy as np
import pytest

from tests.conftest import ROOT


@pytest.fixture(scope="module")
def inspect_mod():
    spec = importlib.util.spec_from_file_location("inspect_sentences_under_test",
                                                  os.path.join(ROOT, "scripts", "inspect_sentences.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_dataset(tmp_path, with_meta=True):
    d = tmp_path / "gsl_sentences"
    (d / "landmarks").mkdir(parents=True)
    rows = []

    def add(split, signer, label, hands_frac, n):
        fname = f"{split}_{signer}_{len(rows)}.npy"
        arr = np.zeros((60, 126), np.float32)
        arr[: int(60 * hands_frac), :63] = 0.5
        np.save(d / "landmarks" / fname, arr)
        rows.append({"file": fname, "split": split, "signer": signer, "scenario": "health1",
                     "sample": f"p{signer}/s{n}", "label": label, "frames_total": "", "hand_frames": ""})

    for n in range(6):
        add("train", 1, "ΓΕΙΑ ΕΓΩ(1)", 0.9, n)
    add("train", 2, "ΓΕΙΑ ΕΓΩ(2)", 0.9, 0)  # παραλλαγή της ίδιας πρότασης: 1 μόνο δείγμα
    add("dev", 3, "ΓΕΙΑ ΕΓΩ(1)", 0.8, 0)
    add("test", 3, "ΓΕΙΑ ΕΓΩ(1)", 0.2, 1)  # κακή ανίχνευση χεριών
    with open(d / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if with_meta:
        (d / "prep_meta.json").write_text(json.dumps({"frames": 60, "trim_margin": 2}), encoding="utf-8")
    return str(d)


def test_reports_shared_signer_label_variants_and_low_hands(inspect_mod, tmp_path, capsys):
    assert inspect_mod.inspect(_make_dataset(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "'trim_margin': 2" in out
    assert "Ίδιος signer σε dev και test: ['3']" in out
    assert "Ίδιος signer σε train" not in out
    assert "  6 /  0 /  0   ΓΕΙΑ ΕΓΩ(1)" not in out  # το (1) έχει και dev/test δείγματα
    assert "  6 /  1 /  1   ΓΕΙΑ ΕΓΩ(1)" in out and "  1 /  0 /  0   ΓΕΙΑ ΕΓΩ(2)" in out
    assert "train ανά label: min 1, max 6" in out
    assert "αρχεία με χέρι σε < 50% των frames: 1 από 9" in out


def test_warns_without_meta_and_fails_without_manifest(inspect_mod, tmp_path, capsys):
    assert inspect_mod.inspect(_make_dataset(tmp_path, with_meta=False)) == 0
    assert "Λείπει το prep_meta.json" in capsys.readouterr().out
    assert inspect_mod.inspect(str(tmp_path / "nothing")) == 1
    assert "prepare_sentences.py" in capsys.readouterr().out
