"""
SignBridge - ρυθμίσεις του backend.

Όλες οι τιμές έχουν προεπιλογή και μπορούν να αλλάξουν με μεταβλητές περιβάλλοντος
SIGNBRIDGE_<ΟΝΟΜΑ> (βλ. .env.example). Το Settings ελέγχει τις τιμές κατά τη δημιουργία του,
ώστε ένα λάθος στο .env να φαίνεται αμέσως και όχι στη μέση μιας σύνδεσης.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Features ανά frame: 2 χέρια x 21 σημεία x (x, y, z).
FEATURE_SIZE = 126
# Μήκος παραθύρου των μοντέλων ΛΕΞΕΩΝ (train.py). Τα μοντέλα προτάσεων γράφουν το δικό τους seq_len στο checkpoint.
MODEL_SEQ_LEN = 30

# "gsl_sentences" -> weights/gsl_sentences_model.pth (ml/training/train_sentences.py)
DEFAULT_LANGUAGES = ("gsl", "asl", "gsl_sentences")

ENV_PREFIX = "SIGNBRIDGE_"
# Εναλλακτικά ονόματα μεταβλητών (το .env.example γράφει SIGNBRIDGE_LANDMARKER_MODEL, όχι ..._PATH).
ENV_ALIASES = {"landmarker_model_path": ("LANDMARKER_MODEL",)}


def _csv(value: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in value.split(",") if p.strip())


@dataclass
class Settings:
    # --- Αρχεία ---
    weights_dir: Path = PROJECT_ROOT / "services" / "inference" / "weights"
    landmarker_model_path: Path = PROJECT_ROOT / "services" / "inference" / "hand_landmarker.task"
    languages: tuple[str, ...] = DEFAULT_LANGUAGES

    # --- Sliding window (μοντέλα λέξεων) ---
    window_frames: int = 30
    min_frames: int = 15
    stride: int = 3
    min_hand_ratio: float = 0.4
    idle_reset_frames: int = 8

    # --- Απόφαση / υπότιτλοι ---
    threshold: float = 0.70
    vote_window: int = 5
    min_votes: int = 3

    # --- Μοντέλα προτάσεων: το inference τρέχει όταν τελειώνει η πρόταση (κατεβαίνουν τα χέρια) ---
    sentence_end_frames: int = 10     # frames χωρίς χέρια = τέλος πρότασης (~0.7 s στα 15 fps)
    sentence_max_frames: int = 225    # ανώτατο μήκος πρότασης (~15 s στα 15 fps)· μετά κόβεται αναγκαστικά
    sentence_min_frames: int = 10     # λιγότερα frames με χέρι = θόρυβος, αγνοείται
    sentence_margin: int = 2          # frames-περιθώριο γύρω από την πρόταση (αν το checkpoint δεν ορίζει δικό του)

    # --- Όρια ---
    max_sessions: int = 2
    max_frame_bytes: int = 2_000_000
    max_frame_width: int = 640
    torch_threads: int = 2

    # --- CORS ---
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.weights_dir = Path(self.weights_dir)
        self.landmarker_model_path = Path(self.landmarker_model_path)
        self.languages = tuple(self.languages)
        self.cors_origins = tuple(self.cors_origins)

        if not self.languages:
            raise ValueError("languages: χρειάζεται τουλάχιστον μία γλώσσα")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold πρέπει να είναι στο (0, 1], βρέθηκε {self.threshold}")
        if not 0.0 <= self.min_hand_ratio <= 1.0:
            raise ValueError(f"min_hand_ratio πρέπει να είναι στο [0, 1], βρέθηκε {self.min_hand_ratio}")
        for name in ("window_frames", "min_frames", "stride", "idle_reset_frames", "vote_window", "min_votes",
                     "sentence_end_frames", "sentence_max_frames", "sentence_min_frames",
                     "max_sessions", "max_frame_bytes", "max_frame_width", "torch_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} πρέπει να είναι >= 1, βρέθηκε {getattr(self, name)}")
        if self.sentence_margin < 0:
            raise ValueError(f"sentence_margin πρέπει να είναι >= 0, βρέθηκε {self.sentence_margin}")
        if self.min_frames > self.window_frames:
            raise ValueError(f"min_frames ({self.min_frames}) > window_frames ({self.window_frames})")
        if self.min_votes > self.vote_window:
            raise ValueError(f"min_votes ({self.min_votes}) > vote_window ({self.vote_window})")
        if self.sentence_min_frames > self.sentence_max_frames:
            raise ValueError(f"sentence_min_frames ({self.sentence_min_frames}) > "
                             f"sentence_max_frames ({self.sentence_max_frames})")

    @classmethod
    def from_env(cls, environ=None) -> "Settings":
        """Διαβάζει τα SIGNBRIDGE_* από το περιβάλλον. Άγνωστες/κενές μεταβλητές αγνοούνται."""
        env = os.environ if environ is None else environ
        kwargs: dict = {}
        for f in fields(cls):
            names = (f.name.upper(),) + ENV_ALIASES.get(f.name, ())
            var, raw = next(((ENV_PREFIX + n, env[ENV_PREFIX + n]) for n in names
                             if (env.get(ENV_PREFIX + n) or "").strip()), (None, None))
            if raw is None:
                continue
            raw = raw.strip()
            try:
                if f.name in ("weights_dir", "landmarker_model_path"):
                    kwargs[f.name] = Path(raw)
                elif f.name in ("languages", "cors_origins"):
                    kwargs[f.name] = _csv(raw)
                elif isinstance(f.default, bool):  # δεν υπάρχει σήμερα bool, αλλά να μη σπάσει αν προστεθεί
                    kwargs[f.name] = raw.lower() in ("1", "true", "yes", "on")
                elif isinstance(f.default, int):
                    kwargs[f.name] = int(raw)
                elif isinstance(f.default, float):
                    kwargs[f.name] = float(raw)
            except ValueError as exc:
                raise ValueError(f"{var}={raw!r} δεν είναι έγκυρη τιμή") from exc
        return cls(**kwargs)
