"""
ml/training/train_sentences.py: leave-one-signer-out (--loso) και τελικό μοντέλο σε όλους τους signers (--final).
Συνθετικά δεδομένα, χωρίς κάμερα, MediaPipe ή το CERTH dataset.

Dataset: 4 signers x 2 προτάσεις (A, B) x 3 επαναλήψεις + μία «σπάνια» πρόταση (RARE) που τη νοημάτισε ΜΟΝΟ ο signer 4
(όπως οι παραλλαγές gloss του πραγματικού dataset): στο fold του signer 4 δεν υπάρχει στην εκπαίδευση, άρα δεν μπορεί
να αναγνωριστεί.
"""
import csv
import json
import os

import numpy as np
import pytest
import torch

from services.config import Settings
from services.inference.live_inference import ModelRegistry

SEQ_LEN = 60
TRIM = 2
LEVELS = {"A": 0.15, "B": 0.85, "RARE": 0.5}
LABELS = {"A": "ΓΕΙΑ ΕΓΩ(1)", "B": "ΕΥΧΑΡΙΣΤΩ ΠΟΛΥ", "RARE": "ΠΑΡΑΛΛΑΓΗ ΜΟΝΟ ΣΤΟΝ ΤΕΤΑΡΤΟ"}
SIGNERS = [1, 2, 3, 4]
SPLIT_OF = {1: "train", 2: "train", 3: "dev", 4: "test"}  # το loso αγνοεί τα splits


def _window(level, rng=None, noise=0.0):
    seq = np.zeros((SEQ_LEN, 126), dtype=np.float32)
    hand = np.array([[level, level + 0.02 * k / 21, level * 0.1] for k in range(21)]).flatten()
    seq[TRIM:SEQ_LEN - TRIM, :63] = hand
    if rng is not None and noise:
        seq[TRIM:SEQ_LEN - TRIM, :63] += rng.normal(0, noise, size=(SEQ_LEN - 2 * TRIM, 63))
    return seq


def make_dataset(root):
    data_dir = root / "gsl_sentences"
    (data_dir / "landmarks").mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = []

    def add(signer, key, rep):
        fname = f"health1_signer{signer}_rep{rep}_sentences_s0000_{key}.npy"
        np.save(data_dir / "landmarks" / fname, _window(LEVELS[key], rng, 0.005))
        rows.append({"file": fname, "split": SPLIT_OF[signer], "signer": signer, "scenario": "health1",
                     "sample": f"health1_signer{signer}_rep{rep}_sentences/sentences0000",
                     "label": LABELS[key], "frames_total": 80, "hand_frames": 70})

    for signer in SIGNERS:
        for key in ("A", "B"):
            for rep in range(3):
                add(signer, key, rep)
    for rep in range(2):
        add(4, "RARE", rep)

    with open(data_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (data_dir / "prep_meta.json").write_text(json.dumps({"frames": SEQ_LEN, "trim_margin": TRIM}), encoding="utf-8")
    return str(data_dir), rows


@pytest.fixture(scope="module")
def ts():
    from ml.training import train_sentences
    return train_sentences


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("loso_data"))


@pytest.fixture(scope="module")
def loso_run(ts, dataset, tmp_path_factory):
    """Τρέχει ΜΙΑ φορά το loso, καταγράφοντας το μέγεθος του training set κάθε fold."""
    data_dir, rows = dataset
    report = tmp_path_factory.mktemp("loso_report") / "report.json"
    seen = []
    original = ts._fit

    def spy(Xtr, ytr, *args, **kwargs):
        seen.append(len(ytr))
        return original(Xtr, ytr, *args, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(ts, "_fit", spy)
    try:
        result = ts.loso(epochs=40, data_dir=data_dir, report_path=str(report), curve_every=20)
    finally:
        mp.undo()
    return result, seen, report, rows


# ------------------------------------------------------------------ βοηθητικά
def test_split_by_signer_is_disjoint_and_complete(ts, dataset):
    _, rows = dataset
    for s in SIGNERS:
        train_rows, held = ts.split_by_signer(rows, s)
        assert held and {int(r["signer"]) for r in held} == {s}
        assert s not in {int(r["signer"]) for r in train_rows}
        assert len(train_rows) + len(held) == len(rows)


def test_signers_of_and_all_rows(ts, dataset):
    data_dir, rows = dataset
    assert ts.signers_of(rows) == SIGNERS
    assert len(ts.all_rows(ts.load_manifest(data_dir))) == len(rows)  # train + dev + test, χωρίς απώλειες


def test_threshold_rows_counts_shown_and_correct(ts):
    conf = [0.95, 0.55, 0.45, 0.85]
    correct = [True, True, False, False]
    rows = {r["threshold"]: (r["shown"], r["correct"], r["n"]) for r in ts.threshold_rows(conf, correct)}
    assert rows[0.3] == (4, 2, 4)
    assert rows[0.5] == (3, 2, 4)  # το 0.45 δεν περνά
    assert rows[0.7] == (2, 1, 4)
    assert rows[0.9] == (1, 1, 4)
    assert ts.threshold_rows([], [])[0]["shown"] == 0  # χωρίς δείγματα δεν σκάει


# ----------------------------------------------------------------------- loso
def test_every_signer_is_held_out_exactly_once(loso_run):
    result, seen, _, rows = loso_run
    assert [f["signer"] for f in result["folds"]] == SIGNERS
    assert sum(f["n"] for f in result["folds"]) == len(rows)  # κάθε δείγμα ελέγχεται μία φορά
    # Το training set κάθε fold = όλα τα δείγματα εκτός από του signer που ελέγχεται.
    per_signer = {s: sum(1 for r in rows if int(r["signer"]) == s) for s in SIGNERS}
    assert seen == [len(rows) - per_signer[s] for s in SIGNERS]
    assert [f["n_train"] for f in result["folds"]] == seen


def test_unreachable_sentence_is_counted_wrong_and_reported(loso_run):
    result, _, _, _ = loso_run
    fold4 = next(f for f in result["folds"] if f["signer"] == 4)
    assert fold4["n"] == 8 and fold4["unreachable"] == 2 and fold4["reachable_n"] == 6
    assert fold4["correct"] <= 6  # οι 2 μη προσβάσιμες δεν μπορούν να είναι σωστές
    assert all(f["unreachable"] == 0 for f in result["folds"] if f["signer"] != 4)
    assert result["unreachable"] == 2
    assert result["pooled_acc"] <= (result["samples"] - 2) / result["samples"] + 1e-9
    assert result["per_label"][LABELS["RARE"]] == {"n": 2, "correct": 0}


def test_loso_learns_separable_signs_in_unseen_signers(loso_run):
    result, _, _, _ = loso_run
    assert result["pooled_reachable_acc"] >= 0.9
    assert result["mean_acc"] == pytest.approx(np.mean([f["acc"] for f in result["folds"]]))
    assert result["min_acc"] <= result["mean_acc"] <= result["max_acc"]
    assert result["std_acc"] >= 0.0


def test_loso_report_file_and_curve(loso_run):
    result, _, report, _ = loso_run
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["signers"] == SIGNERS and data["epochs"] == 40
    assert set(data["curve_mean"]) == {"20", "40"}  # curve_every=20, και πάντα η τελευταία εποχή
    assert {t["threshold"] for t in data["thresholds"]} >= {0.5, 0.7}
    assert all(0.0 <= t["correct"] / t["shown"] <= 1.0 for t in data["thresholds"] if t["shown"])
    assert LABELS["A"] in data["per_label"]
    # Στο JSON τα κλειδιά της καμπύλης είναι strings: ίδιες εποχές με το αποτέλεσμα στη μνήμη.
    assert [set(map(int, f["curve"])) for f in data["folds"]] == [set(f["curve"]) for f in result["folds"]]


def test_loso_needs_at_least_two_signers(ts, tmp_path):
    data_dir, rows = make_dataset(tmp_path)
    kept = [r for r in rows if int(r["signer"]) == 1]
    with open(os.path.join(data_dir, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    with pytest.raises(RuntimeError, match="2 signers"):
        ts.loso(epochs=1, data_dir=data_dir)


# ---------------------------------------------------------------------- final
@pytest.fixture(scope="module")
def final_ckpt(ts, dataset, tmp_path_factory):
    data_dir, _ = dataset
    out = tmp_path_factory.mktemp("final_weights") / "gsl_sentences_model.pth"
    ts.train_final(epochs=40, data_dir=data_dir, out_path=str(out), loso_acc=0.875)
    return out


def test_final_checkpoint_loads_in_server_registry(final_ckpt):
    reg = ModelRegistry(final_ckpt.parent, Settings().languages)
    m = reg.get("gsl_sentences")
    assert m.mode == "sentence" and m.seq_len == SEQ_LEN and m.trim_margin == TRIM
    assert m.labels == sorted(LABELS.values())  # η ένωση ΟΛΩΝ των προτάσεων, και της σπάνιας
    for key in ("A", "B"):
        probs = m.predict_proba(_window(LEVELS[key]))
        assert m.labels[int(probs.argmax())] == LABELS[key]
        assert probs.max() > 0.5
    assert reg.describe()["gsl_sentences"]["available"] is True


def test_final_checkpoint_records_what_it_was_trained_on(final_ckpt):
    ckpt = torch.load(final_ckpt, weights_only=True)  # ο server φορτώνει με weights_only=True
    assert ckpt["trained_on"] == "all_signers" and ckpt["signers"] == SIGNERS
    assert ckpt["epochs"] == 40 and ckpt["loso_acc"] == pytest.approx(0.875)
    assert "test_acc" not in ckpt and "dev_acc" not in ckpt  # δεν υπάρχει ανεξάρτητο test: δεν ψεύδεται η αναφορά


def test_final_backs_up_the_previous_model(ts, dataset, tmp_path):
    data_dir, _ = dataset
    out = tmp_path / "gsl_sentences_model.pth"
    out.write_bytes(b"old-model-bytes")
    ts.train_final(epochs=1, data_dir=data_dir, out_path=str(out))
    backups = [p for p in tmp_path.iterdir() if "_prev_" in p.name]
    assert len(backups) == 1 and backups[0].read_bytes() == b"old-model-bytes"
    assert out.read_bytes() != b"old-model-bytes"
    # Το αντίγραφο ασφαλείας δεν ταιριάζει με το όνομα που διαβάζει ο server.
    assert ModelRegistry(tmp_path, Settings().languages).weights_path("gsl_sentences") == out

    ts.train_final(epochs=1, data_dir=data_dir, out_path=str(out))  # δεύτερο τρέξιμο: δεύτερο, διαφορετικό backup
    assert len([p for p in tmp_path.iterdir() if "_prev_" in p.name]) == 2


def test_backup_existing_returns_none_without_file(ts, tmp_path):
    assert ts.backup_existing(str(tmp_path / "missing.pth")) is None


# ------------------------------------------------------------------------ CLI
def test_cli_loso_and_final_together(ts, dataset, tmp_path):
    data_dir, _ = dataset
    out, report = tmp_path / "w" / "gsl_sentences_model.pth", tmp_path / "r.json"
    code = ts.main(["--loso", "--final", "--epochs", "2", "--data-dir", data_dir,
                    "--out", str(out), "--report", str(report)])
    assert code == 0 and report.is_file() and out.is_file()
    loso_acc = json.loads(report.read_text(encoding="utf-8"))["pooled_acc"]
    assert torch.load(out, weights_only=True)["loso_acc"] == pytest.approx(loso_acc)  # το --final κρατά το αποτέλεσμα του --loso


def test_cli_final_alone_has_no_loso_acc(ts, dataset, tmp_path):
    data_dir, _ = dataset
    out = tmp_path / "gsl_sentences_model.pth"
    assert ts.main(["--final", "--epochs", "1", "--data-dir", data_dir, "--out", str(out)]) == 0
    assert torch.load(out, weights_only=True)["loso_acc"] is None
