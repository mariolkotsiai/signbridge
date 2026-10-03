"""
Έλεγχος του LandmarkExtractor.detect() με το ΠΡΑΓΜΑΤΙΚΟ mediapipe API (mp.Image, Options),
αλλά με ψεύτικο HandLandmarker, ώστε να μη χρειάζεται το hand_landmarker.task.
ΔΕΝ αντικαθιστά δοκιμή με πραγματικό μοντέλο και πραγματικά χέρια.
"""
import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

from services.inference import landmarks_pipeline as lp  # noqa: E402


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []
        self.images = []

    def detect_for_video(self, mp_image, ts):
        self.images.append(mp_image)
        self.timestamps.append(ts)
        return "RESULT"

    def close(self):
        pass


@pytest.fixture()
def extractor(tmp_path, monkeypatch):
    fake = FakeLandmarker()
    monkeypatch.setattr(mp_vision.HandLandmarker, "create_from_options", staticmethod(lambda opts: fake))
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"x")
    ext = lp.LandmarkExtractor(model_path=str(model))
    return ext, fake


def test_detect_returns_result_without_touching_frame(extractor):
    ext, fake = extractor
    frame = np.full((48, 64, 3), 7, dtype=np.uint8)
    before = frame.copy()
    assert ext.detect(frame) == "RESULT"
    assert np.array_equal(frame, before)  # καμία σχεδίαση πάνω στο frame
    assert fake.images[0].width == 64 and fake.images[0].height == 48


def test_timestamps_are_strictly_increasing_even_in_same_millisecond(extractor):
    ext, fake = extractor
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(200):  # πολύ πιο γρήγορα από 1 ms/frame
        ext.detect(frame)
    ts = fake.timestamps
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))
