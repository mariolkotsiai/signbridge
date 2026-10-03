"""
Ρυθμίσεις (configuration) για την υπηρεσία συμπερασμού (inference) του SignBridge.

Όλες οι τιμές μπορούν να αλλάξουν με μεταβλητές περιβάλλοντος SIGNBRIDGE_*
(βλ. .env.example), χωρίς αλλαγή κώδικα.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "services" / "inference"

# Πρέπει να ταιριάζει με το input_size του SignLanguageLSTM (2 χέρια * 21 σημεία * 3 συντεταγμένες).
FEATURE_SIZE = 126
# Το μήκος που περιμένει το μοντέλο (ίδιο με το target_frames του normalize_sequence).
MODEL_SEQ_LEN = 30


def _env(name: str, default: T, cast: Callable[[str], T]) -> T:
    """Διαβάζει SIGNBRIDGE_<name> από το περιβάλλον, με έλεγχο τύπου."""
    raw = os.environ.get(f"SIGNBRIDGE_{name}")
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Μη έγκυρη τιμή για SIGNBRIDGE_{name}: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # --- Αρχεία ---
    weights_dir: Path = INFERENCE_DIR / "weights"
    landmarker_model_path: Path = INFERENCE_DIR / "hand_landmarker.task"
    languages: tuple[str, ...] = ("gsl", "asl")

    # --- Buffering / ρυθμός inference ---
    window_frames: int = 30        # μέγιστο μήκος sliding window (σε frames που στέλνει ο client)
    min_frames: int = 15           # ελάχιστο μήκος για να τρέξει το πρώτο inference
    stride: int = 3                # τρέχει inference κάθε N νέα frames
    min_hand_ratio: float = 0.4    # ελάχιστο ποσοστό frames του window με ανιχνευμένο χέρι
    idle_reset_frames: int = 8     # N συνεχόμενα frames χωρίς χέρι => IDLE (καθαρίζει buffer)

    # --- Απόφαση / σταθεροποίηση υποτίτλων ---
    threshold: float = 0.70        # ελάχιστη εμπιστοσύνη για να γίνει δεκτή μια πρόβλεψη
    vote_window: int = 5           # πόσες τελευταίες προβλέψεις συμμετέχουν στην ψηφοφορία
    min_votes: int = 3             # πόσες ίδιες ψήφοι χρειάζονται για "σταθερό" label

    # --- Όρια / ασφάλεια ---
    max_sessions: int = 2
    max_frame_bytes: int = 2_000_000
    max_frame_width: int = 640     # μεγαλύτερα frames κάνουν downscale πριν το MediaPipe
    torch_threads: int = 2

    # --- CORS (μόνο για απευθείας κλήσεις από άλλο origin, π.χ. Vite χωρίς proxy) ---
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    def __post_init__(self) -> None:
        if not (1 <= self.min_frames <= self.window_frames):
            raise ValueError("Πρέπει 1 <= min_frames <= window_frames")
        if self.stride < 1:
            raise ValueError("Το stride πρέπει να είναι >= 1")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError("Το threshold πρέπει να είναι στο (0, 1]")
        if not (1 <= self.min_votes <= self.vote_window):
            raise ValueError("Πρέπει 1 <= min_votes <= vote_window")
        if self.max_sessions < 1:
            raise ValueError("Το max_sessions πρέπει να είναι >= 1")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()  # προεπιλογές
        origins = _env("CORS_ORIGINS", None, lambda s: tuple(o.strip() for o in s.split(",") if o.strip()))
        langs = _env("LANGUAGES", None, lambda s: tuple(x.strip().lower() for x in s.split(",") if x.strip()))
        return cls(
            weights_dir=_env("WEIGHTS_DIR", d.weights_dir, Path),
            landmarker_model_path=_env("LANDMARKER_MODEL", d.landmarker_model_path, Path),
            languages=langs or d.languages,
            window_frames=_env("WINDOW_FRAMES", d.window_frames, int),
            min_frames=_env("MIN_FRAMES", d.min_frames, int),
            stride=_env("STRIDE", d.stride, int),
            min_hand_ratio=_env("MIN_HAND_RATIO", d.min_hand_ratio, float),
            idle_reset_frames=_env("IDLE_RESET_FRAMES", d.idle_reset_frames, int),
            threshold=_env("THRESHOLD", d.threshold, float),
            vote_window=_env("VOTE_WINDOW", d.vote_window, int),
            min_votes=_env("MIN_VOTES", d.min_votes, int),
            max_sessions=_env("MAX_SESSIONS", d.max_sessions, int),
            max_frame_bytes=_env("MAX_FRAME_BYTES", d.max_frame_bytes, int),
            max_frame_width=_env("MAX_FRAME_WIDTH", d.max_frame_width, int),
            torch_threads=_env("TORCH_THREADS", d.torch_threads, int),
            cors_origins=origins or d.cors_origins,
        )
