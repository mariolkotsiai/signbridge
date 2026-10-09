"""
Λειτουργία προτάσεων: checkpoint του train_sentences.py -> registry -> session ("τέλος πρότασης") -> API,
και τα βοηθητικά replay/προετοιμασίας. Όλα με συνθετικά δεδομένα και ψεύτικο extractor
(δεν χρειάζεται κάμερα, MediaPipe μοντέλο ή το CERTH dataset).
"""
import csv
import importlib.util
import json
import os

import cv2
import numpy as np
import pytest
import torch

from services.config import PROJECT_ROOT, Settings
from services.inference.live_inference import (
    InferenceSession,
    ModelLoadError,
    ModelRegistry,
    iter_png_frames,
    load_manifest_samples,
    replay_samples,
    summarize_outcomes,
)
from tests.conftest import LEVELS, FakeExtractor
from tests.test_api import build_client, stream, wait_for

LABEL_A = "ΓΕΙΑ ΕΓΩ(1)"
LABEL_B = "ΕΥΧΑΡΙΣΤΩ ΠΟΛΥ"
SENTENCES = {LABEL_A: "A", LABEL_B: "B"}
SEQ_LEN = 60
TRIM = 2
BLANK = np.zeros((48, 64, 3), np.uint8)


def _sample(level: float, rng) -> np.ndarray:
    """Συνθετική πρόταση όπως βγαίνει από το prepare_sentences (περιθώριο TRIM frames χωρίς χέρια)."""
    seq = np.zeros((SEQ_LEN, 126), dtype=np.float32)
    hand = np.array([[level, level + 0.02 * k / 21, level * 0.1] for k in range(21)]).flatten()
    seq[TRIM:SEQ_LEN - TRIM, :63] = hand + rng.normal(0, 0.005, size=(SEQ_LEN - 2 * TRIM, 63))
    return seq


@pytest.fixture(scope="session")
def sentence_weights(tmp_path_factory):
    """Εκπαιδεύει ΠΡΑΓΜΑΤΙΚΑ με το ml/training/train_sentences.py σε συνθετικό dataset (2 προτάσεις, 5 signers)."""
    from ml.training import train_sentences

    root = tmp_path_factory.mktemp("sentences")
    data_dir = root / "data"
    (data_dir / "landmarks").mkdir(parents=True)
    rng = np.random.default_rng(7)
    rows = []
    for split, signers in {"train": [1, 2, 3], "dev": [4], "test": [5]}.items():
        for signer in signers:
            for label, key in SENTENCES.items():
                for rep in range(3):
                    fname = f"health1_signer{signer}_rep{rep}_sentences_sentences0000_{key}.npy"
                    np.save(data_dir / "landmarks" / fname, _sample(LEVELS[key], rng))
                    rows.append({"file": fname, "split": split, "signer": signer, "scenario": "health1",
                                 "sample": f"health1_signer{signer}_rep{rep}_sentences/sentences0000",
                                 "label": label, "frames_total": 80, "hand_frames": 70})
    with open(data_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (data_dir / "prep_meta.json").write_text(json.dumps({"frames": SEQ_LEN, "trim_margin": TRIM}), encoding="utf-8")

    weights = root / "weights"
    train_sentences.train(epochs=30, data_dir=str(data_dir), out_path=str(weights / "gsl_sentences_model.pth"))
    assert (weights / "gsl_sentences_model.pth").is_file()
    return weights


@pytest.fixture()
def sentence_settings(sentence_weights):
    return Settings(weights_dir=sentence_weights)


def make_session(settings, script, model=None):
    model = model or ModelRegistry(settings.weights_dir, settings.languages).get("gsl_sentences")
    ext = FakeExtractor(script)
    return InferenceSession("gsl_sentences", model, ext, settings), ext


def run(session, n):
    results = [session.process_frame(BLANK) for _ in range(n)]
    committed = [r["prediction"]["stable_label"] for r in results
                 if r["prediction"] and r["prediction"]["committed"]]
    return results, committed


def hands_then_none(n_hands, n_none, sign="A"):
    return lambda i: sign if i < n_hands else None


class StubSentenceModel:
    """Μοντέλο που επιστρέφει σταθερές πιθανότητες και θυμάται το τελευταίο παράθυρο που είδε."""
    mode = "sentence"
    seq_len = SEQ_LEN
    trim_margin = None
    labels = ["P", "Q"]

    def __init__(self, probs=(0.6, 0.4)):
        self.probs = np.array(probs, dtype=np.float32)
        self.last_window = None

    def predict_proba(self, window):
        self.last_window = window
        return self.probs


# ----------------------------------------------------------------- registry
def test_checkpoint_from_train_sentences_has_new_fields(sentence_settings):
    reg = ModelRegistry(sentence_settings.weights_dir, sentence_settings.languages)
    m = reg.get("gsl_sentences")
    assert m.mode == "sentence" and m.seq_len == SEQ_LEN and m.trim_margin == TRIM
    assert m.labels == sorted([LABEL_A, LABEL_B])
    info = reg.describe()["gsl_sentences"]
    assert info["available"] and info["mode"] == "sentence" and info["seq_len"] == SEQ_LEN
    assert reg.describe()["gsl"]["available"] is False  # δεν υπάρχει μοντέλο λέξεων σε αυτόν τον φάκελο


def test_word_checkpoint_keeps_old_defaults(settings):
    m = ModelRegistry(settings.weights_dir).get("gsl")  # γραμμένο από το train.py, χωρίς τα νέα πεδία
    assert m.mode == "sign" and m.seq_len == 30 and m.trim_margin is None


def test_mode_is_inferred_from_name_for_old_sentence_checkpoints(sentence_settings, tmp_path):
    ckpt = torch.load(sentence_settings.weights_dir / "gsl_sentences_model.pth", weights_only=True)
    for key in ("mode", "trim_margin"):
        ckpt.pop(key)
    torch.save(ckpt, tmp_path / "gsl_sentences_model.pth")
    m = ModelRegistry(tmp_path).get("gsl_sentences")
    assert m.mode == "sentence" and m.seq_len == SEQ_LEN and m.trim_margin is None


@pytest.mark.parametrize("patch", [{"mode": "banana"}, {"seq_len": 1}, {"seq_len": "abc"}])
def test_invalid_checkpoint_fields_are_rejected(sentence_settings, tmp_path, patch):
    ckpt = torch.load(sentence_settings.weights_dir / "gsl_sentences_model.pth", weights_only=True)
    ckpt.update(patch)
    torch.save(ckpt, tmp_path / "gsl_sentences_model.pth")
    with pytest.raises(ModelLoadError):
        ModelRegistry(tmp_path).get("gsl_sentences")


# ------------------------------------------------------------------ session
def test_commits_one_subtitle_when_hands_leave(sentence_settings):
    s, _ = make_session(sentence_settings, hands_then_none(40, 0))
    end = sentence_settings.sentence_end_frames
    results, committed = run(s, 40 + end + 12)

    assert committed == [LABEL_A]
    # Κανένα inference όσο φαίνονται χέρια: μόνο συλλογή.
    during = results[:40]
    assert all(r["state"] == "collecting" and r["prediction"] is None for r in during)
    assert [r["buffer"]["filled"] for r in during][:3] == [1, 2, 3] and during[-1]["buffer"]["filled"] == 40
    assert during[-1]["buffer"]["size"] == sentence_settings.sentence_max_frames

    # Το inference τρέχει ακριβώς στο `end`-οστό frame χωρίς χέρια.
    final_idx = 40 + end - 1
    final = results[final_idx]
    assert final["prediction"]["final"] and final["prediction"]["committed"]
    assert final["prediction"]["confidence"] > 0.7
    assert final["state"] == "sign" and final["stable_label"] == LABEL_A and final["buffer"]["filled"] == 0
    assert all(r["prediction"] is None for i, r in enumerate(results) if i != final_idx)

    # Το αποτέλεσμα μένει λίγο στην οθόνη και μετά πάμε idle.
    assert results[final_idx + 1]["state"] == "sign"
    assert results[-1]["state"] == "idle" and results[-1]["stable_label"] is None


def test_two_sentences_separated_by_a_pause(sentence_settings):
    script = lambda i: "A" if i < 40 else (None if i < 60 else ("B" if i < 100 else None))  # noqa: E731
    s, _ = make_session(sentence_settings, script)
    _, committed = run(s, 130)
    assert committed == [LABEL_A, LABEL_B]


def test_short_detection_gap_does_not_split_a_sentence(sentence_settings):
    script = lambda i: None if 20 <= i < 25 else ("A" if i < 45 else None)  # noqa: E731
    s, _ = make_session(sentence_settings, script)
    results, committed = run(s, 80)
    assert committed == [LABEL_A]
    assert sum(1 for r in results if r["prediction"]) == 1


def test_noise_is_ignored(sentence_settings):
    s, _ = make_session(sentence_settings, hands_then_none(3, 0))  # 3 frames με χέρι < sentence_min_frames
    results, committed = run(s, 40)
    assert committed == [] and all(r["prediction"] is None for r in results)
    assert results[-1]["state"] == "idle" and results[-1]["buffer"]["filled"] == 0


def test_hands_that_never_leave_are_cut_at_max_length(sentence_settings):
    cfg = Settings(**{**sentence_settings.__dict__, "sentence_max_frames": 50})
    s, _ = make_session(cfg, lambda i: "A")
    results, committed = run(s, 130)
    assert committed == [LABEL_A] * 2  # frames 50 και 100
    assert max(r["buffer"]["filled"] for r in results) <= 50


def test_below_threshold_is_uncertain_and_not_committed(sentence_settings):
    stub = StubSentenceModel((0.6, 0.4))  # threshold 0.70
    s, _ = make_session(sentence_settings, hands_then_none(30, 0), model=stub)
    results, committed = run(s, 30 + sentence_settings.sentence_end_frames)
    final = results[-1]
    assert committed == []
    assert final["state"] == "uncertain" and final["stable_label"] is None
    assert final["prediction"]["accepted"] is False and final["prediction"]["label"] == "P"

    s.configure(threshold=0.5)  # χαμηλώνει το όριο: η επόμενη πρόταση γίνεται αποδεκτή
    s.process_frame(BLANK)  # (τα μαύρα frames δεν έχουν χέρι: μένουμε σε αναμονή)
    s2, _ = make_session(sentence_settings, hands_then_none(30, 0), model=StubSentenceModel((0.6, 0.4)))
    s2.configure(threshold=0.5)
    _, committed2 = run(s2, 30 + sentence_settings.sentence_end_frames)
    assert committed2 == ["P"]


def test_window_has_model_seq_len_and_zero_margins(sentence_settings):
    stub = StubSentenceModel((0.9, 0.1))
    s, _ = make_session(sentence_settings, hands_then_none(30, 0), model=stub)
    run(s, 30 + sentence_settings.sentence_end_frames)
    w = stub.last_window
    assert w.shape == (SEQ_LEN, 126)  # seq_len του μοντέλου, όχι το σταθερό 30
    assert not w[0].any() and not w[-1].any()  # περιθώριο χωρίς χέρια (sentence_margin=2)
    assert w[SEQ_LEN // 2].any()


def test_margin_comes_from_checkpoint_when_present(sentence_settings):
    stub = StubSentenceModel((0.9, 0.1))
    stub.trim_margin = 0
    s, _ = make_session(sentence_settings, hands_then_none(30, 0), model=stub)
    run(s, 30 + sentence_settings.sentence_end_frames)
    assert stub.last_window[0].any()  # χωρίς περιθώριο: το πρώτο frame έχει χέρι


def test_reset_discards_sentence_in_progress(sentence_settings):
    s, _ = make_session(sentence_settings, lambda i: "A")
    run(s, 20)
    s.reset()
    out = s.process_frame(BLANK)  # νέο χέρι: ξεκινά καινούργια πρόταση με 1 frame
    assert out["buffer"]["filled"] == 1 and out["stable_label"] is None


def test_ready_message_describes_sentence_mode(sentence_settings):
    s, _ = make_session(sentence_settings, None)
    msg = s.ready_message()
    assert msg["mode"] == "sentence" and msg["seq_len"] == SEQ_LEN
    assert msg["window_frames"] == sentence_settings.sentence_max_frames
    assert msg["sentence_end_frames"] == sentence_settings.sentence_end_frames


# ---------------------------------------------------------------------- API
def test_health_and_websocket_with_sentence_model(sentence_settings, jpeg_bytes):
    client, service, extractors = build_client(sentence_settings, hands_then_none(40, 0))
    with client:
        health = client.get("/api/health").json()
        assert health["models"]["gsl_sentences"]["available"] is True
        assert health["models"]["gsl_sentences"]["mode"] == "sentence"

        with client.websocket_connect("/ws/infer?language=gsl_sentences") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["mode"] == "sentence"
            msgs = stream(ws, jpeg_bytes, 40 + sentence_settings.sentence_end_frames + 2)
        assert wait_for(lambda: service.active_sessions == 0)
    commits = [m["prediction"] for m in msgs if m["prediction"] and m["prediction"]["committed"]]
    assert [c["stable_label"] for c in commits] == [LABEL_A]


def test_predict_sequence_uses_model_seq_len(sentence_settings):
    client, _, _ = build_client(sentence_settings)
    level = LEVELS["A"]
    row = [v for k in range(21) for v in (level, level + 0.02 * k / 21, level * 0.1)] + [0.0] * 63
    with client:
        r = client.post("/api/predict/sequence", json={"language": "gsl_sentences", "sequence": [row] * 25})
    assert r.status_code == 200 and r.json()["label"] == LABEL_A


# ----------------------------------------------------------------- settings
def test_new_settings_validation_and_env():
    with pytest.raises(ValueError):
        Settings(sentence_end_frames=0)
    with pytest.raises(ValueError):
        Settings(sentence_min_frames=300)  # > sentence_max_frames
    with pytest.raises(ValueError):
        Settings(sentence_margin=-1)
    cfg = Settings.from_env({"SIGNBRIDGE_SENTENCE_END_FRAMES": "7", "SIGNBRIDGE_THRESHOLD": "0.55",
                             "SIGNBRIDGE_LANGUAGES": "gsl, gsl_sentences"})
    assert cfg.sentence_end_frames == 7 and cfg.threshold == 0.55 and cfg.languages == ("gsl", "gsl_sentences")
    with pytest.raises(ValueError, match="SIGNBRIDGE_STRIDE"):
        Settings.from_env({"SIGNBRIDGE_STRIDE": "abc"})
    # Το όνομα του .env.example (χωρίς _PATH) και το πλήρες όνομα του πεδίου δουλεύουν και τα δύο.
    assert Settings.from_env({"SIGNBRIDGE_LANDMARKER_MODEL": "x/h.task"}).landmarker_model_path.as_posix() == "x/h.task"
    assert Settings.from_env({"SIGNBRIDGE_LANDMARKER_MODEL_PATH": "y/h.task"}).landmarker_model_path.as_posix() == "y/h.task"
    assert Settings.from_env({"SIGNBRIDGE_STRIDE": "  "}).stride == Settings().stride  # κενό = προεπιλογή


# ------------------------------------------------------------------- replay
def test_replay_samples_reports_accuracy(sentence_settings):
    script = lambda i: "A" if i < 40 else (None if i < 60 else ("B" if i < 100 else None))  # noqa: E731
    s, _ = make_session(sentence_settings, script)
    samples = [{"name": "s1", "label": LABEL_A, "frames": [BLANK] * 40},
               {"name": "s2", "label": LABEL_A, "frames": [BLANK] * 40}]  # το 2ο «νοηματίζει» B => λάθος
    outcomes = replay_samples(s, samples)
    assert [o["predicted"] for o in outcomes] == [LABEL_A, LABEL_B]
    assert all(o["accepted"] for o in outcomes)
    report = summarize_outcomes(outcomes)
    assert "[OK]" in report and "[ΛΑΘΟΣ]" in report and "1/2" in report


def test_threshold_table_counts_shown_and_correct():
    from services.inference.live_inference import threshold_table

    outcomes = [
        {"expected": "A", "predicted": "A", "confidence": 0.95},
        {"expected": "A", "predicted": "A", "confidence": 0.55},
        {"expected": "A", "predicted": "B", "confidence": 0.45},   # λάθος με χαμηλή εμπιστοσύνη
        {"expected": "A", "predicted": "B", "confidence": 0.85},   # λάθος με ψηλή εμπιστοσύνη
        {"expected": "A", "predicted": None, "confidence": 0.0},   # καμία απόφαση: δεν εμφανίζεται ποτέ
    ]
    rows = {}
    for line in threshold_table(outcomes).splitlines():
        if "|" in line and "%" in line.split("|")[0]:
            head, shown, correct = (c.strip() for c in line.split("|"))
            rows[head] = (shown, correct)
    assert rows == {
        "30%": ("4/5", "2/4 = 50%"),
        "40%": ("4/5", "2/4 = 50%"),   # το 0.45 περνά ακόμη
        "50%": ("3/5", "2/3 = 67%"),
        "60%": ("2/5", "1/2 = 50%"),
        "70%": ("2/5", "1/2 = 50%"),
        "80%": ("2/5", "1/2 = 50%"),
        "90%": ("1/5", "1/1 = 100%"),
    }
    assert threshold_table([]) and "-" in threshold_table([])  # χωρίς δείγματα δεν σκάει (ούτε διαιρεί με 0)


def test_replay_without_hands_gives_no_decision(sentence_settings):
    s, _ = make_session(sentence_settings, lambda i: None)
    out = replay_samples(s, [{"name": "empty", "label": LABEL_A, "frames": [BLANK] * 30}])[0]
    assert out["predicted"] is None
    assert "ΧΩΡΙΣ ΑΠΟΦΑΣΗ" in summarize_outcomes([out])


def test_replay_subsample_keeps_every_nth_frame(sentence_settings):
    s, ext = make_session(sentence_settings, lambda i: None)
    replay_samples(s, [{"name": "x", "frames": [BLANK] * 80}], subsample=2)
    flush = sentence_settings.sentence_end_frames + sentence_settings.idle_reset_frames + 2
    assert ext.calls == 40 + flush


def test_png_frames_in_natural_order_with_greek_path(tmp_path):
    folder = tmp_path / "δεδομένα" / "ΓΕΙΑ"
    folder.mkdir(parents=True)
    for n in (10, 2, 1):  # το 2 πρέπει να προηγείται του 10
        img = np.full((8, 8, 3), n, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(folder / f"{n}.png"))  # tofile: λειτουργεί και με μη-ASCII διαδρομή
    frames = list(iter_png_frames(str(folder)))
    assert [int(f[0, 0, 0]) for f in frames] == [1, 2, 10]


def test_manifest_samples_alternate_signers_and_report_missing(tmp_path):
    root = tmp_path / "root"
    rows = []
    for signer in (5, 6):
        for rep in range(3):
            parent = f"health1_signer{signer}_rep{rep}_sentences"
            if not (signer == 6 and rep == 2):  # ένα δείγμα λείπει από το δίσκο
                (root / "health1" / parent / "sentences0000").mkdir(parents=True)
            rows.append({"file": f"{parent}.npy", "split": "test", "signer": signer, "scenario": "health1",
                         "sample": f"{parent}/sentences0000", "label": LABEL_A, "frames_total": "", "hand_frames": ""})
    (root / "health1" / "health1_signer1_rep0_sentences" / "sentences0000").mkdir(parents=True)
    rows.append({**rows[0], "split": "train", "signer": 1, "sample": "health1_signer1_rep0_sentences/sentences0000"})
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Ζητάμε 6: ο κύκλος φτάνει στο signer6/rep2 που λείπει από το δίσκο, το παραλείπει και το μετράει.
    picked, missing = load_manifest_samples(str(manifest), str(root), count=6, split="test")
    assert [p["signer"] for p in picked] == ["5", "6", "5", "6", "5"]  # εναλλαγή signers
    assert missing == 1 and all(os.path.isdir(p["path"]) for p in picked)
    assert all(p["label"] == LABEL_A for p in picked)

    # Με λιγότερα από όσα υπάρχουν, επιστρέφει ακριβώς όσα ζητήθηκαν και εναλλάσσει signers.
    few, none_missing = load_manifest_samples(str(manifest), str(root), count=2, split="test")
    assert [p["signer"] for p in few] == ["5", "6"] and none_missing == 0
    # Το split φιλτράρεται: από το train υπάρχει ένα μόνο δείγμα.
    assert len(load_manifest_samples(str(manifest), str(root), count=5, split="train")[0]) == 1


# ----------------------------------------------------- scripts/prepare_sentences
@pytest.fixture(scope="module")
def prep():
    pytest.importorskip("mediapipe")  # το script εισάγει τον LandmarkExtractor
    path = os.path.join(PROJECT_ROOT, "scripts", "prepare_sentences.py")
    spec = importlib.util.spec_from_file_location("prepare_sentences_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kp(active: bool):
    return np.ones(126, dtype=np.float32) if active else np.zeros(126, dtype=np.float32)


def test_trim_sequence_keeps_margin_around_hands(prep):
    seq = [_kp(False)] * 6 + [_kp(True)] * 4 + [_kp(False)] * 7
    out = prep.trim_sequence(seq, 2)
    assert len(out) == 4 + 2 + 2
    assert not out[0].any() and not out[-1].any() and out[2].any() and out[5].any()
    assert len(prep.trim_sequence(seq, None)) == len(seq) and len(prep.trim_sequence(seq, -1)) == len(seq)
    assert len(prep.trim_sequence([_kp(False)] * 5, 2)) == 5  # χωρίς χέρι πουθενά: ως έχει
    assert len(prep.trim_sequence(seq, 100)) == len(seq)  # το περιθώριο δεν βγαίνει εκτός ορίων


def test_prepare_cache_is_renamed_when_settings_change(prep, tmp_path):
    out = str(tmp_path / "gsl_sentences")
    meta = {"frames": 60, "trim_margin": 2}
    assert prep._prepare_cache(out, meta) is False  # πρώτη φορά: δεν υπάρχει cache
    np.save(os.path.join(out, "landmarks", "a.npy"), np.zeros((60, 126), np.float32))

    assert prep._prepare_cache(out, meta) is True  # ίδιες ρυθμίσεις: επαναχρησιμοποίηση
    assert os.path.exists(os.path.join(out, "landmarks", "a.npy"))

    assert prep._prepare_cache(out, {"frames": 60, "trim_margin": None}) is False  # άλλαξε το κόψιμο
    assert not os.listdir(os.path.join(out, "landmarks"))
    stale = [d for d in os.listdir(out) if d.startswith("landmarks_stale_")]
    assert len(stale) == 1 and os.path.exists(os.path.join(out, stale[0], "a.npy"))  # δεν χάθηκαν δεδομένα
    with open(os.path.join(out, "prep_meta.json"), encoding="utf-8") as f:
        assert json.load(f) == {"frames": 60, "trim_margin": None}


def test_server_window_equals_training_window(prep, sentence_settings):
    """
    Train/serve skew: η ΙΔΙΑ αλληλουχία landmarks πρέπει να καταλήγει στο ΙΔΙΟ παράθυρο, είτε περάσει από το
    prepare_sentences.py (εκπαίδευση: trim_sequence + normalize_sequence) είτε από το session του server.
    """
    from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence
    from tests.conftest import FakeExtractor

    script = lambda i: "A" if 5 <= i < 35 else None  # noqa: E731  5 κενά, 30 με χέρι, μετά κενά
    end = sentence_settings.sentence_end_frames

    stub = StubSentenceModel((0.9, 0.1))
    stub.trim_margin = TRIM
    s, _ = make_session(sentence_settings, script, model=stub)
    run(s, 35 + end)
    server_window = stub.last_window
    assert server_window is not None and server_window.shape == (SEQ_LEN, 126)

    # Πλευρά εκπαίδευσης: τα ίδια frames (ο ψεύτικος extractor είναι ντετερμινιστικός), όπως τα βλέπει το prepare.
    ref = FakeExtractor(script)
    frames = [extract_hand_keypoints(ref.detect(BLANK)).astype(np.float32) for _ in range(35 + end)]
    train_window = normalize_sequence(prep.trim_sequence(frames, TRIM), target_frames=SEQ_LEN).astype(np.float32)

    assert train_window.shape == server_window.shape
    np.testing.assert_allclose(server_window, train_window, atol=1e-6)
