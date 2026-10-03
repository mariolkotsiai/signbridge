"""
Μετατρέπει δείγματα νοημάτων σε .npy αρχεία landmarks (30 frames x 126 τιμές) για το training.

Κάθε δείγμα μπορεί να είναι:
  - αρχείο βίντεο (.mp4 / .avi / .mov)
  - φάκελος με εικόνες (.png / .jpg), π.χ. οι ακολουθίες PNG του CERTH GSL dataset

Είσοδος:  ml/raw_videos/<γλώσσα>/<ΛΕΞΗ>/<δείγμα>
Έξοδος:   ml/datasets/<γλώσσα>/raw_landmarks/<ΛΕΞΗ>_open_<δείγμα>.npy

Χρήση:
    python scripts\\prepare_open_dataset.py gsl
    python scripts\\prepare_open_dataset.py gsl --max-per-label 30     (γρήγορη δοκιμή)

Το script είναι επαναλήψιμο: δείγματα που έχουν ήδη μετατραπεί παραλείπονται,
οπότε αν το διακόψεις (Ctrl+C) συνεχίζει από εκεί που έμεινε.
"""
import argparse
import os
import re
import sys
import time

import cv2
import numpy as np

# Προσθήκη του root directory στο PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.inference.landmarks_pipeline import LandmarkExtractor
from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence

VIDEO_EXTS = (".mp4", ".avi", ".mov")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _natural_key(name: str):
    """frame_2 πριν από frame_10 (φυσική ταξινόμηση αντί για αλφαβητική)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def read_image(path: str):
    """
    Διαβάζει εικόνα ακόμη και αν η διαδρομή έχει ελληνικούς χαρακτήρες.
    Το cv2.imread αποτυγχάνει (επιστρέφει None) με μη-ASCII διαδρομές σε Windows,
    και εδώ οι φάκελοι ονομάζονται με τη λέξη (π.χ. ...\\ΓΕΙΑ\\glosses0000\\...).
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def iter_frames(sample_path: str):
    """Παράγει τα frames (BGR) ενός δείγματος: φάκελος εικόνων ή αρχείο βίντεο."""
    if os.path.isdir(sample_path):
        names = sorted(
            (f for f in os.listdir(sample_path) if f.lower().endswith(IMAGE_EXTS)),
            key=_natural_key,
        )
        for name in names:
            frame = read_image(os.path.join(sample_path, name))
            if frame is not None:
                yield frame
    else:
        cap = cv2.VideoCapture(sample_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()


def detect_hands(extractor: LandmarkExtractor, frame):
    """Προτιμά το detect() (αύξοντα timestamps, χωρίς σχεδίαση)· αλλιώς το παλιό process_frame()."""
    detect = getattr(extractor, "detect", None)
    return detect(frame) if detect else extractor.process_frame(frame)[1]


def process_sample(sample_path: str, extractor: LandmarkExtractor):
    """Επιστρέφει (πίνακας 30x126, frames με χέρι, σύνολο frames)."""
    sequence, with_hands = [], 0
    for frame in iter_frames(sample_path):
        keypoints = extract_hand_keypoints(detect_hands(extractor, frame))
        if keypoints.any():
            with_hands += 1
        sequence.append(keypoints)
    return normalize_sequence(sequence, target_frames=30), with_hands, len(sequence)


def process_video_file(video_path: str, extractor: LandmarkExtractor) -> np.ndarray:
    """Συμβατότητα με παλιό κώδικα: μόνο ο πίνακας 30x126."""
    return process_sample(video_path, extractor)[0]


def list_samples(label_dir: str):
    """Δείγματα μιας λέξης: υποφάκελοι (ακολουθίες εικόνων) και αρχεία βίντεο."""
    out = []
    for name in os.listdir(label_dir):
        path = os.path.join(label_dir, name)
        if os.path.isdir(path) or name.lower().endswith(VIDEO_EXTS):
            out.append(name)
    return sorted(out, key=_natural_key)


def convert_dataset_videos(language="gsl", max_per_label=None, min_hand_ratio=0.2):
    raw_dir = os.path.join(PROJECT_ROOT, f"ml/raw_videos/{language}")
    out_dir = os.path.join(PROJECT_ROOT, f"ml/datasets/{language}/raw_landmarks")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος: {raw_dir}")
        return

    converted = skipped_existing = skipped_empty = 0
    started = time.time()

    for label in sorted(os.listdir(raw_dir)):
        label_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(label_dir):
            continue

        samples = list_samples(label_dir)
        if max_per_label:
            samples = samples[:max_per_label]
        if not samples:
            continue
        print(f"\n[INFO] '{label}': {len(samples)} δείγματα")

        for idx, name in enumerate(samples, 1):
            sample_path = os.path.join(label_dir, name)
            sample_id = name if os.path.isdir(sample_path) else os.path.splitext(name)[0]
            save_path = os.path.join(out_dir, f"{label}_open_{sample_id}.npy")

            if os.path.exists(save_path):
                skipped_existing += 1
                continue

            # Νέος extractor ανά δείγμα: το MediaPipe (VIDEO mode) κρατά tracking από το
            # προηγούμενο frame, και δεν θέλουμε να «περνάει» το χέρι από το ένα δείγμα στο άλλο.
            extractor = LandmarkExtractor()
            try:
                data, with_hands, total = process_sample(sample_path, extractor)
            finally:
                extractor.close()

            if total == 0 or with_hands / total < min_hand_ratio:
                print(f"  [SKIP] {name}: χέρια σε {with_hands}/{total} frames (όριο {min_hand_ratio:.0%})")
                skipped_empty += 1
                continue

            np.save(save_path, data.astype(np.float32))
            converted += 1
            print(f"  [{idx}/{len(samples)}] {name}: {total} frames, χέρια σε {with_hands} -> {os.path.basename(save_path)}")

    minutes = (time.time() - started) / 60
    print(f"\n[SUCCESS] Ολοκληρώθηκε σε {minutes:.1f} λεπτά.")
    print(f" -> Νέα: {converted} | ήδη υπήρχαν: {skipped_existing} | χωρίς χέρια: {skipped_empty}")
    print(f" -> Έξοδος: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Μετατροπή δειγμάτων σε landmarks (.npy)")
    ap.add_argument("language", nargs="?", default="gsl", choices=["gsl", "asl"])
    ap.add_argument("--max-per-label", type=int, default=None,
                    help="μέγιστος αριθμός δειγμάτων ανά λέξη (για γρήγορη δοκιμή)")
    ap.add_argument("--min-hand-ratio", type=float, default=0.2,
                    help="ελάχιστο ποσοστό frames με χέρι για να κρατηθεί το δείγμα (προεπιλογή 0.2)")
    args = ap.parse_args()
    convert_dataset_videos(args.language, args.max_per_label, args.min_hand_ratio)