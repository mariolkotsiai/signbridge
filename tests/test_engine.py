import numpy as np
import pytest
import torch

from ml.models.sign_model import SignLanguageLSTM
from services.config import Settings
from services.inference.live_inference import (
    BadFrame,
    InferenceSession,
    ModelLoadError,
    ModelNotAvailable,
    ModelRegistry,
    SubtitleStabilizer,
)
from tests.conftest import LEVELS, FakeExtractor


# ----------------------------------------------------------------- registry
def test_loads_checkpoint_written_by_train_py(settings):
    reg = ModelRegistry(settings.weights_dir)
    m = reg.get("gsl")
    assert m.labels == ["A", "B"]
    # Τα features του fake «A» πρέπει να αναγνωρίζονται με υψηλή εμπιστοσύνη.
    window = np.zeros((30, 126), dtype=np.float32)
    window[:, :63] = np.array([[LEVELS["A"], LEVELS["A"] + 0.02 * k / 21, LEVELS["A"] * 0.1]
                               for k in range(21)]).flatten()
    probs = m.predict_proba(window)
    assert m.labels[int(probs.argmax())] == "A"
    assert probs.max() > 0.9


def test_registry_caches_and_reloads_on_change(settings, tmp_path):
    reg = ModelRegistry(settings.weights_dir)
    assert reg.get("gsl") is reg.get("gsl")
    reg.reload()
    assert reg.get("gsl") is not None


def test_missing_model_raises(settings):
    with pytest.raises(ModelNotAvailable):
        ModelRegistry(settings.weights_dir).get("asl")


def test_corrupt_file_raises(tmp_path):
    (tmp_path / "gsl_model.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(ModelLoadError):
        ModelRegistry(tmp_path).get("gsl")


def test_wrong_feature_size_is_rejected(tmp_path):
    bad = SignLanguageLSTM(num_classes=2, input_size=225)  # π.χ. εκπαίδευση με Holistic
    torch.save({"model_state": bad.state_dict(), "labels": ["X", "Y"]}, tmp_path / "gsl_model.pth")
    with pytest.raises(ModelLoadError, match="225"):
        ModelRegistry(tmp_path).get("gsl")


def test_hyperparameters_are_inferred_from_weights(tmp_path):
    custom = SignLanguageLSTM(num_classes=3, hidden_size=64, num_layers=1)
    torch.save({"model_state": custom.state_dict(), "labels": ["a", "b", "c"]}, tmp_path / "gsl_model.pth")
    m = ModelRegistry(tmp_path).get("gsl")
    assert m.predict_proba(np.zeros((30, 126), dtype=np.float32)).shape == (3,)


# --------------------------------------------------------------- stabilizer
def test_stabilizer_commits_once_then_waits_for_idle():
    s = SubtitleStabilizer(vote_window=5, min_votes=3)
    outs = [s.update("X") for _ in range(6)]
    assert [o.committed for o in outs] == [False, False, True, False, False, False]
    s.notify_idle()
    assert [s.update("X").committed for _ in range(3)] == [False, False, True]


def test_stabilizer_ignores_low_confidence_votes():
    s = SubtitleStabilizer(5, 3)
    for lbl in ["X", None, "X", None, None]:
        out = s.update(lbl)
    assert out.stable_label is None and not out.committed


def test_stabilizer_switches_label():
    s = SubtitleStabilizer(5, 3)
    committed = [o.stable_label for o in (s.update(l) for l in ["X"] * 4 + ["Y"] * 5) if o.committed]
    assert committed == ["X", "Y"]


# ------------------------------------------------------------------ session
def make_session(settings, script, language="gsl"):
    model = ModelRegistry(settings.weights_dir).get(language)
    ext = FakeExtractor(script)
    return InferenceSession(language, model, ext, settings), ext


def run(session, n):
    results = [session.process_frame(np.zeros((48, 64, 3), np.uint8)) for _ in range(n)]
    committed = [r["prediction"]["stable_label"] for r in results
                 if r["prediction"] and r["prediction"]["committed"]]
    return results, committed


def test_starts_idle_and_buffers_nothing_without_hands(settings):
    s, _ = make_session(settings, lambda i: None)
    results, committed = run(s, 20)
    assert {r["state"] for r in results} == {"idle"}
    assert results[-1]["buffer"]["filled"] == 0 and committed == []


def test_recognises_sign_and_emits_exactly_one_subtitle(settings):
    s, _ = make_session(settings, lambda i: "A")
    results, committed = run(s, 60)
    assert committed == ["A"]
    assert results[-1]["state"] == "sign" and results[-1]["stable_label"] == "A"
    pred = next(r["prediction"] for r in results if r["prediction"])
    assert set(pred) >= {"label", "confidence", "accepted", "top_k", "stable_label", "committed"}
    assert all(0 <= v <= 1 for h in results[-1]["hands"] for pt in h for v in pt)


def test_collecting_until_min_frames(settings):
    s, _ = make_session(settings, lambda i: "A")
    results, _ = run(s, settings.min_frames - 1)
    assert all(r["state"] == "collecting" and r["prediction"] is None for r in results)


def test_same_sign_repeats_after_hands_leave(settings):
    script = lambda i: "A" if (i < 50 or i >= 70) else None  # noqa: E731
    s, _ = make_session(settings, script)
    results, committed = run(s, 130)
    assert committed == ["A", "A"]
    assert any(r["state"] == "idle" for r in results[50:75])


def test_switching_signs_without_lowering_hands(settings):
    s, _ = make_session(settings, lambda i: "A" if i < 50 else "B")
    _, committed = run(s, 130)
    assert committed[0] == "A" and committed[-1] == "B"


def test_uncertain_below_threshold_and_threshold_is_adjustable(settings):
    class Stub:
        labels = ["P", "Q"]
        def predict_proba(self, window):  # noqa: E301
            return np.array([0.6, 0.4], dtype=np.float32)

    ext = FakeExtractor(lambda i: "A")
    s = InferenceSession("gsl", Stub(), ext, settings)  # threshold 0.70
    results, committed = run(s, 40)
    assert committed == [] and results[-1]["state"] == "uncertain"
    assert results[-1]["prediction"] is None or results[-1]["prediction"]["accepted"] is False

    s.configure(threshold=0.5)
    _, committed = run(s, 20)
    assert committed == ["P"]


def test_reset_clears_state(settings):
    s, _ = make_session(settings, lambda i: "A")
    run(s, 40)
    s.reset()
    out = s.process_frame(np.zeros((48, 64, 3), np.uint8))
    assert out["buffer"]["filled"] == 1 and out["stable_label"] is None


# --------------------------------------------------------------- JPEG input
def test_bad_jpeg_raises(settings):
    s, _ = make_session(settings, None)
    with pytest.raises(BadFrame):
        s.process_jpeg(b"definitely not an image")
    with pytest.raises(BadFrame):
        s.process_jpeg(b"")


def test_large_frames_are_downscaled(settings, jpeg_bytes):
    import cv2
    s, ext = make_session(settings, None)
    big = cv2.imencode(".jpg", np.zeros((960, 1920, 3), np.uint8))[1].tobytes()
    s.process_jpeg(big)
    assert ext.last_shape[1] <= settings.max_frame_width
    s.process_jpeg(jpeg_bytes)
    assert ext.last_shape[:2] == (240, 320)


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(min_frames=40, window_frames=30)
    with pytest.raises(ValueError):
        Settings(min_votes=6, vote_window=5)
    with pytest.raises(ValueError):
        Settings(threshold=0)
