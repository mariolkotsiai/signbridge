"""
SignBridge - Live inference engine.

Αυτό το module ΔΕΝ εξαρτάται από το FastAPI. Περιέχει όλη τη λογική που μετατρέπει
μια ροή frames σε υπότιτλους:

    JPEG bytes -> decode -> MediaPipe landmarks -> sliding buffer
               -> normalize_sequence (30 frames) -> LSTM -> softmax
               -> threshold -> ψηφοφορία (debounce) -> committed υπότιτλος

Βασικά κομμάτια:
    ModelRegistry       φορτώνει/κάνει cache τα {gsl,asl}_model.pth (ένα ανά γλώσσα)
    InferenceSession    η κατάσταση ΜΙΑΣ σύνδεσης (buffer, tracker χεριών, ψηφοφορία)
    InferenceService    ανοίγει/κλείνει sessions, εφαρμόζει όριο ταυτόχρονων συνδέσεων
    SubtitleStabilizer  αποτρέπει το "τρεμόπαιγμα" των labels

Χρήση χωρίς browser (debug με κάμερα ή βίντεο):
    python -m services.inference.live_inference --language gsl --show
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import cv2
import numpy as np
import torch

# Εξασφάλιση ότι το root του project είναι στο sys.path (όπως στα υπόλοιπα scripts του repo).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.sign_model import SignLanguageLSTM  # noqa: E402
from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence  # noqa: E402
from services.config import FEATURE_SIZE, MODEL_SEQ_LEN, Settings  # noqa: E402

log = logging.getLogger("signbridge.inference")


# --------------------------------------------------------------------------- #
# Σφάλματα
# --------------------------------------------------------------------------- #
class ModelNotAvailable(Exception):
    """Δεν υπάρχει αρχείο βαρών για τη ζητούμενη γλώσσα."""


class ModelLoadError(Exception):
    """Το αρχείο βαρών υπάρχει αλλά είναι άκυρο/ασύμβατο."""


class LandmarkerUnavailable(Exception):
    """Λείπει το hand_landmarker.task ή το mediapipe."""


class TooManySessions(Exception):
    """Έχει συμπληρωθεί το όριο ταυτόχρονων συνδέσεων."""


class BadFrame(ValueError):
    """Το payload δεν είναι έγκυρη εικόνα JPEG/PNG."""


class LandmarkDetector(Protocol):
    """Ό,τι χρειάζεται το session από έναν extractor (το LandmarkExtractor τα ικανοποιεί)."""

    def detect(self, frame: np.ndarray) -> Any: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Μοντέλο + registry
# --------------------------------------------------------------------------- #
def format_top_k(probs: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    """Επιστρέφει τις k πιο πιθανές κλάσεις ως [{label, confidence}, ...]."""
    k = max(1, min(k, len(labels)))
    order = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in order]


@dataclass
class LoadedModel:
    language: str
    model: SignLanguageLSTM
    labels: list[str]
    path: Path
    mtime: float

    @torch.inference_mode()
    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """window: (30, 126) -> διάνυσμα πιθανοτήτων (num_classes,)."""
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0)
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


class ModelRegistry:
    """
    Φορτώνει τα εκπαιδευμένα βάρη που αποθηκεύει το ml/training/train.py:
        {'model_state': state_dict, 'labels': [...]}  ->  <weights_dir>/<language>_model.pth

    Τα hyper-parameters (hidden_size, num_layers, input_size) εξάγονται από τα ίδια τα
    βάρη, οπότε αλλαγές στην αρχιτεκτονική του SignLanguageLSTM δεν σπάνε τη φόρτωση.
    Αν το αρχείο αλλάξει στο δίσκο (νέα εκπαίδευση), το επόμενο get() το ξαναφορτώνει.
    """

    def __init__(self, weights_dir: Path, languages: tuple[str, ...] = ("gsl", "asl")):
        self.weights_dir = Path(weights_dir)
        self.languages = languages
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def weights_path(self, language: str) -> Path:
        return self.weights_dir / f"{language}_model.pth"

    def get(self, language: str) -> LoadedModel:
        path = self.weights_path(language)
        if not path.is_file():
            raise ModelNotAvailable(
                f"Δεν βρέθηκε μοντέλο για '{language}': {path}. "
                f"Εκπαίδευσέ το με: python -m ml.training.train"
            )
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(language)
            if cached is not None and cached.mtime == mtime:
                return cached
            loaded = self._load(language, path, mtime)
            self._cache[language] = loaded
            log.info("Φορτώθηκε μοντέλο %s: %d κλάσεις %s", language, len(loaded.labels), loaded.labels)
            return loaded

    def reload(self) -> None:
        """Αδειάζει το cache - τα επόμενα sessions θα φορτώσουν ξανά από το δίσκο."""
        with self._lock:
            self._cache.clear()

    def describe(self) -> dict[str, dict]:
        """Κατάσταση κάθε γλώσσας (για /api/health). Δεν πετάει exceptions."""
        out: dict[str, dict] = {}
        for lang in self.languages:
            path = self.weights_path(lang)
            try:
                m = self.get(lang)
                out[lang] = {"available": True, "num_classes": len(m.labels),
                             "labels": m.labels, "path": str(path)}
            except ModelNotAvailable as exc:
                out[lang] = {"available": False, "error": str(exc), "path": str(path)}
            except ModelLoadError as exc:
                out[lang] = {"available": False, "error": str(exc), "path": str(path)}
        return out

    @staticmethod
    def _load(language: str, path: Path, mtime: float) -> LoadedModel:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt["model_state"]
            labels = [str(x) for x in ckpt["labels"]]
        except Exception as exc:  # κατεστραμμένο αρχείο, λάθος format κ.λπ.
            raise ModelLoadError(f"Αδυναμία ανάγνωσης {path.name}: {exc}") from exc

        try:
            input_size = state["lstm.weight_ih_l0"].shape[1]
            hidden_size = state["lstm.weight_hh_l0"].shape[1]
            num_layers = sum(1 for k in state if k.startswith("lstm.weight_ih_l") and not k.endswith("_reverse"))
            num_classes = state["fc.weight"].shape[0]
        except KeyError as exc:
            raise ModelLoadError(f"Το {path.name} δεν ταιριάζει με το SignLanguageLSTM (λείπει {exc}).") from exc

        if input_size != FEATURE_SIZE:
            raise ModelLoadError(
                f"Το {path.name} έχει input_size={input_size}, αλλά το live pipeline παράγει {FEATURE_SIZE} "
                f"features (2 χέρια x 21 x 3). Αν εκπαιδεύτηκε με Holistic (225 features), "
                f"ξαναεκπαίδευσέ το από landmarks του prepare_open_dataset.py."
            )
        if num_classes != len(labels):
            raise ModelLoadError(f"Το {path.name}: {num_classes} έξοδοι αλλά {len(labels)} labels.")

        model = SignLanguageLSTM(num_classes=num_classes, input_size=input_size,
                                 hidden_size=hidden_size, num_layers=num_layers)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(f"Ασύμβατα βάρη στο {path.name}: {exc}") from exc
        model.eval()
        return LoadedModel(language, model, labels, path, mtime)


# --------------------------------------------------------------------------- #
# Σταθεροποίηση υποτίτλων
# --------------------------------------------------------------------------- #
@dataclass
class StabilizerOutput:
    stable_label: Optional[str]
    committed: bool


class SubtitleStabilizer:
    """
    Ψηφοφορία πάνω στις τελευταίες `vote_window` προβλέψεις. Ένα label γίνεται "σταθερό"
    όταν συγκεντρώσει >= `min_votes` ψήφους. Εκδίδεται (committed=True) ΜΙΑ φορά, και η ίδια
    λέξη δεν ξαναεκδίδεται μέχρι να αλλάξει το label ή να φύγουν τα χέρια (notify_idle).
    Οι προβλέψεις κάτω από το threshold μετράνε ως ψήφος None.
    """

    def __init__(self, vote_window: int, min_votes: int):
        self.vote_window = vote_window
        self.min_votes = min_votes
        self._votes: deque[Optional[str]] = deque(maxlen=vote_window)
        self._committed: Optional[str] = None
        self.stable_label: Optional[str] = None

    def update(self, label: Optional[str]) -> StabilizerOutput:
        self._votes.append(label)
        counts = Counter(v for v in self._votes if v is not None)
        stable = None
        if counts:
            top, n = counts.most_common(1)[0]
            if n >= self.min_votes:
                stable = top
        committed = stable is not None and stable != self._committed
        if committed:
            self._committed = stable
        self.stable_label = stable
        return StabilizerOutput(stable, committed)

    def notify_idle(self) -> None:
        """Τα χέρια εξαφανίστηκαν: επιτρέπει να ξαναεμφανιστεί η ίδια λέξη."""
        self._votes.clear()
        self._committed = None
        self.stable_label = None

    reset = notify_idle


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class InferenceSession:
    """
    Κατάσταση μίας σύνδεσης. ΔΕΝ είναι thread-safe: καλείται διαδοχικά από ένα μόνο
    consumer (ο WebSocket handler περιμένει κάθε frame πριν διαβάσει το επόμενο).

    Καταστάσεις (state) που βλέπει το UI:
        idle        δεν φαίνονται χέρια
        collecting  γεμίζει ο buffer / λίγα frames με χέρι
        sign        υπάρχει σταθερή αναγνώριση (stable_label)
        uncertain   έτρεξε inference αλλά η εμπιστοσύνη < threshold
    """

    def __init__(self, language: str, model: LoadedModel, extractor: LandmarkDetector, settings: Settings):
        self.language = language
        self.model = model
        self.settings = settings
        self.threshold = settings.threshold
        self._extractor = extractor
        self._stabilizer = SubtitleStabilizer(settings.vote_window, settings.min_votes)
        self._buffer: deque[np.ndarray] = deque(maxlen=settings.window_frames)
        self._hand_flags: deque[bool] = deque(maxlen=settings.window_frames)
        self._seq = 0
        self._last_frame_t: Optional[float] = None
        self._fps = 0.0
        # Το close() δεν πρέπει να τρέξει όσο επεξεργάζεται frame σε άλλο thread.
        self._lock = threading.Lock()
        self._closed = False
        self.reset()

    # -- δημόσιο API -------------------------------------------------------- #
    def ready_message(self) -> dict:
        s = self.settings
        return {
            "type": "ready", "language": self.language, "labels": self.model.labels,
            "window_frames": s.window_frames, "min_frames": s.min_frames, "stride": s.stride,
            "threshold": self.threshold, "min_votes": self._stabilizer.min_votes,
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._hand_flags.clear()
        self._stabilizer.reset()
        self._no_hand_streak = self.settings.idle_reset_frames  # ξεκινά "idle" μέχρι να φανεί χέρι
        self._since_infer = 0
        self._state = "idle"

    def configure(self, threshold: Optional[float] = None, min_votes: Optional[int] = None) -> None:
        if threshold is not None:
            self.threshold = float(min(max(threshold, 0.05), 0.999))
        if min_votes is not None:
            self._stabilizer.min_votes = int(min(max(min_votes, 1), self._stabilizer.vote_window))

    def runtime_config(self) -> dict:
        return {"threshold": self.threshold, "min_votes": self._stabilizer.min_votes}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._extractor.close()
            except Exception:  # pragma: no cover - best effort
                log.exception("Σφάλμα κατά το κλείσιμο του extractor")

    def process_jpeg(self, data: bytes) -> dict:
        """Αποκωδικοποιεί ένα JPEG/PNG και καλεί το process_frame. Πετάει BadFrame."""
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if frame is None:
            raise BadFrame("Το frame δεν αποκωδικοποιείται ως εικόνα.")
        max_w = self.settings.max_frame_width
        if frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return self.process_frame(frame)

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("Το session έχει κλείσει")
            return self._process_frame_locked(frame_bgr)

    def _process_frame_locked(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        s = self.settings
        self._seq += 1

        # 1) Landmarks. Χρησιμοποιούμε ΤΟ ΙΔΙΟ extract_hand_keypoints με το training,
        #    ώστε τα features (σειρά χεριών, τιμές) να είναι πανομοιότυπα.
        result = self._extractor.detect(frame_bgr)
        keypoints = extract_hand_keypoints(result).astype(np.float32)
        hands = _hands_for_overlay(result)
        has_hand = len(hands) > 0

        # 2) Παρακολούθηση παρουσίας χεριών.
        self._no_hand_streak = 0 if has_hand else self._no_hand_streak + 1
        idle = self._no_hand_streak >= s.idle_reset_frames

        prediction: Optional[dict] = None
        if idle:
            if self._buffer or self._stabilizer.stable_label is not None:
                self._buffer.clear()
                self._hand_flags.clear()
                self._since_infer = 0
                self._stabilizer.notify_idle()
            self._state = "idle"
        else:
            if self._state == "idle":
                self._state = "collecting"
            self._buffer.append(keypoints)
            self._hand_flags.append(has_hand)
            self._since_infer += 1

            n = len(self._buffer)
            hand_ratio = sum(self._hand_flags) / max(n, 1)
            if n < s.min_frames or hand_ratio < s.min_hand_ratio:
                self._state = "collecting"
            elif self._since_infer >= s.stride:
                self._since_infer = 0
                prediction = self._run_inference()

        # 3) Στατιστικά και απάντηση.
        now = time.perf_counter()
        if self._last_frame_t is not None and now > self._last_frame_t:
            inst = 1.0 / (now - self._last_frame_t)
            self._fps = inst if self._fps == 0.0 else 0.8 * self._fps + 0.2 * inst
        self._last_frame_t = now

        return {
            "type": "result",
            "seq": self._seq,
            "language": self.language,
            "state": self._state,
            "hands": hands,
            "buffer": {"filled": len(self._buffer), "size": s.window_frames},
            "stable_label": self._stabilizer.stable_label,
            "prediction": prediction,
            "latency_ms": round((now - t0) * 1000.0, 1),
            "fps": round(self._fps, 1),
        }

    # -- εσωτερικά ---------------------------------------------------------- #
    def _run_inference(self) -> dict:
        # Όπως στο training: όλη η αλληλουχία -> 30 frames με ομοιόμορφη δειγματοληψία.
        window = normalize_sequence(list(self._buffer), target_frames=MODEL_SEQ_LEN)
        probs = self.model.predict_proba(window)
        top_k = format_top_k(probs, self.model.labels, k=3)
        best = top_k[0]
        accepted = best["confidence"] >= self.threshold

        out = self._stabilizer.update(best["label"] if accepted else None)
        self._state = "sign" if out.stable_label else "uncertain"
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "accepted": accepted,
            "top_k": top_k,
            "stable_label": out.stable_label,
            "committed": out.committed,
        }


def _hands_for_overlay(result: Any) -> list[list[list[float]]]:
    """[[ [x, y], ... 21 σημεία ], ...] σε κανονικοποιημένες συντεταγμένες 0..1."""
    landmarks = getattr(result, "hand_landmarks", None) or []
    return [[[round(float(lm.x), 4), round(float(lm.y), 4)] for lm in hand] for hand in landmarks]


# --------------------------------------------------------------------------- #
# Service (sessions + όρια)
# --------------------------------------------------------------------------- #
def default_extractor_factory(settings: Settings) -> Callable[[], LandmarkDetector]:
    """Δημιουργεί ΝΕΟ LandmarkExtractor ανά session (το MediaPipe κρατά tracking state)."""

    def factory() -> LandmarkDetector:
        try:
            from services.inference.landmarks_pipeline import LandmarkExtractor
        except ImportError as exc:
            raise LandmarkerUnavailable(f"Δεν φορτώθηκε το mediapipe: {exc}") from exc
        try:
            return LandmarkExtractor(model_path=str(settings.landmarker_model_path))
        except FileNotFoundError as exc:
            raise LandmarkerUnavailable(
                f"Λείπει το {settings.landmarker_model_path}. Τρέξε: python scripts/download_models.py"
            ) from exc

    return factory


class InferenceService:
    def __init__(self, settings: Optional[Settings] = None, registry: Optional[ModelRegistry] = None,
                 extractor_factory: Optional[Callable[[], LandmarkDetector]] = None):
        self.settings = settings or Settings.from_env()
        self.registry = registry or ModelRegistry(self.settings.weights_dir, self.settings.languages)
        self._extractor_factory = extractor_factory or default_extractor_factory(self.settings)
        # Ένα worker ανά πιθανό session: το MediaPipe + PyTorch τρέχουν εκτός event loop.
        self.executor = ThreadPoolExecutor(max_workers=self.settings.max_sessions,
                                           thread_name_prefix="signbridge")
        self._active = 0
        self._lock = threading.Lock()
        torch.set_num_threads(self.settings.torch_threads)

    @property
    def active_sessions(self) -> int:
        return self._active

    def landmarker_available(self) -> bool:
        return Path(self.settings.landmarker_model_path).is_file()

    def open_session_sync(self, language: str) -> InferenceSession:
        model = self.registry.get(language)  # πετάει ModelNotAvailable / ModelLoadError
        with self._lock:
            if self._active >= self.settings.max_sessions:
                raise TooManySessions(f"Όριο ταυτόχρονων συνδέσεων: {self.settings.max_sessions}")
            self._active += 1
        try:
            extractor = self._extractor_factory()
            return InferenceSession(language, model, extractor, self.settings)
        except Exception:
            with self._lock:
                self._active -= 1
            raise

    def _release_slot(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def close_session_sync(self, session: InferenceSession) -> None:
        try:
            session.close()
        finally:
            self._release_slot()

    async def open_session(self, language: str) -> InferenceSession:
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.open_session_sync, language)

    async def close_session(self, session: InferenceSession) -> None:
        fut = asyncio.get_running_loop().run_in_executor(self.executor, session.close)
        try:
            # shield: αν ακυρωθεί ο handler (shutdown), το κλείσιμο του extractor ολοκληρώνεται παρ' όλα αυτά.
            await asyncio.shield(fut)
        finally:
            # Το slot ελευθερώνεται ΠΑΝΤΑ, ακόμη κι αν ο executor έχει ήδη κλείσει / η εργασία ακυρωθεί.
            self._release_slot()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# CLI demo (χωρίς browser): κάμερα ή αρχείο βίντεο
# --------------------------------------------------------------------------- #
def _run_cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SignBridge live inference (desktop demo)")
    ap.add_argument("--language", default="gsl", choices=["gsl", "asl"])
    ap.add_argument("--source", default="0", help="index κάμερας (0) ή διαδρομή βίντεο")
    ap.add_argument("--show", action="store_true", help="άνοιγμα παραθύρου OpenCV (q για έξοδο)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    service = InferenceService()
    try:
        session = service.open_session_sync(args.language)
    except (ModelNotAvailable, ModelLoadError, LandmarkerUnavailable) as exc:
        print(f"[ERROR] {exc}")
        return 1

    source: Any = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Δεν άνοιξε η πηγή: {args.source}")
        service.close_session_sync(session)
        return 1

    print("[INFO] Ξεκίνησε. Ctrl+C ή 'q' για έξοδο.")
    last_printed = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = session.process_frame(frame)
            pred = out["prediction"]
            if pred and pred["committed"]:
                print(f"[SUBTITLE] {pred['stable_label']}  ({pred['confidence']:.0%})")
            elif out["state"] != last_printed:
                print(f"[STATE] {out['state']}")
            last_printed = out["state"]
            if args.show:
                text = f"{out['state']} {out['stable_label'] or ''} {out['fps']:.0f}fps"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("SignBridge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        service.close_session_sync(session)
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
