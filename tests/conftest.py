"""
Κοινά fixtures. Δεν χρειάζεται κάμερα, MediaPipe ή hand_landmarker.task:
ο extractor αντικαθίσταται από ένα ψεύτικο που επιστρέφει προκαθορισμένα landmarks.
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.config import Settings  # noqa: E402

# Δύο "νοήματα" με πολύ διαφορετικά landmarks, ώστε το μικρό LSTM να τα ξεχωρίζει εύκολα.
LEVELS = {"A": 0.15, "B": 0.85}


def make_hand(level: float, jitter: float = 0.0, rng=None):
    """Ένα χέρι: 21 σημεία με x,y,z γύρω από το `level`."""
    rng = rng or np.random.default_rng(0)
    return [
        SimpleNamespace(
            x=float(np.clip(level + rng.normal(0, jitter), 0, 1)),
            y=float(np.clip(level + 0.02 * i / 21 + rng.normal(0, jitter), 0, 1)),
            z=float(level * 0.1),
        )
        for i in range(21)
    ]


class FakeExtractor:
    """Υλοποιεί το πρωτόκολλο LandmarkDetector. `script` καθορίζει τι «βλέπει» σε κάθε frame."""

    def __init__(self, script=None):
        self.script = script  # callable(frame_index) -> "A" | "B" | None
        self.calls = 0
        self.closed = False
        self.last_shape = None
        self._rng = np.random.default_rng(1)

    def detect(self, frame):
        self.last_shape = frame.shape
        sign = self.script(self.calls) if self.script else "A"
        self.calls += 1
        if sign is None:
            return SimpleNamespace(hand_landmarks=[])
        return SimpleNamespace(hand_landmarks=[make_hand(LEVELS[sign], 0.005, self._rng)])

    def close(self):
        self.closed = True


@pytest.fixture(scope="session")
def jpeg_bytes():
    img = np.full((240, 320, 3), 127, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def trained_root(tmp_path_factory):
    """
    Εκπαιδεύει ΠΡΑΓΜΑΤΙΚΑ με το ml/training/train.py (σε συνθετικό dataset, μόνο GSL),
    ώστε να ελέγχεται ότι το format του .pth ταιριάζει με τον loader του server.
    """
    from ml.training import train

    root = tmp_path_factory.mktemp("signbridge_root")
    data_dir = root / "ml" / "datasets" / "gsl" / "raw_landmarks"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for label, level in LEVELS.items():
        for i in range(20):
            seq = np.zeros((30, 126), dtype=np.float32)
            hand = np.array([[level, level + 0.02 * k / 21, level * 0.1] for k in range(21)]).flatten()
            seq[:, :63] = hand + rng.normal(0, 0.005, size=(30, 63))
            np.save(data_dir / f"{label}_open_{i}_{1000 + i}.npy", seq)

    mp = pytest.MonkeyPatch()
    mp.setattr(train, "PROJECT_ROOT", str(root))
    try:
        train.train_model(language="gsl", epochs=40)
    finally:
        mp.undo()

    assert (root / "services" / "inference" / "weights" / "gsl_model.pth").is_file()
    return root


@pytest.fixture()
def settings(trained_root):
    return Settings(weights_dir=trained_root / "services" / "inference" / "weights")
