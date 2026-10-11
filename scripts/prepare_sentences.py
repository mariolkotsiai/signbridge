"""
Προετοιμασία landmarks για αναγνώριση ΠΡΟΤΑΣΕΩΝ από το CERTH GSL (continuous).

Κάθε δείγμα είναι ένας φάκελος με PNG (μία πρόταση). Ως κλάση χρησιμοποιείται η ακολουθία λέξεων
του split αρχείου, π.χ.  "ΓΕΙΑ ΕΓΩ(1) ΜΠΟΡΩ ΒΟΗΘΩ".
Ο διαχωρισμός train / dev / test είναι ΑΝΑ SIGNER (gsl_split_SI_*.csv): το μοντέλο αξιολογείται
σε άτομα που δεν έχει δει στην εκπαίδευση.

Είσοδος:
    <root>\\health1\\health1_signer1_rep1_sentences\\sentences0000\\*.png
    <root>\\GSL_split\\GSL_continuous\\gsl_split_SI_{train,dev,test}.csv

Έξοδος (ml/datasets/gsl_sentences/):
    landmarks/<φάκελος>_<δείγμα>.npy     πίνακας (frames x 126)
    manifest.csv                         αρχείο, split, signer, label, ...

Χρήση:
    python scripts\\prepare_sentences.py --limit 10          (δοκιμαστικά: μετράς την ταχύτητα)
    python scripts\\prepare_sentences.py                      (όλα τα δείγματα του health1)
    python scripts\\prepare_sentences.py --scenarios health1 kep1

Κόψιμο (trim): αφαιρούνται τα frames στην αρχή/τέλος που δεν φαίνεται χέρι (ο signer ξεκινά και τελειώνει
με τα χέρια κάτω), κρατώντας λίγα frames περιθώριο (--trim-margin, προεπιλογή 2). Έτσι τα δεδομένα
εκπαίδευσης μοιάζουν με αυτό που βλέπει ο server: εκεί η πρόταση ξεκινά όταν εμφανιστεί χέρι και τελειώνει
όταν χαθεί. Με --no-trim κρατιέται όλο το βίντεο (δεν συνιστάται για χρήση με τον server).

Επαναλήψιμο: ό,τι έχει ήδη μετατραπεί με τις ΙΔΙΕΣ ρυθμίσεις (--frames, --trim-margin) παραλείπεται.
Αν αλλάξεις ρυθμίσεις, ο παλιός φάκελος landmarks μετονομάζεται σε landmarks_stale_<ώρα> και τα δείγματα
ξαναϋπολογίζονται (μπορείς να τον σβήσεις).
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.inference.landmarks_pipeline import LandmarkExtractor  # noqa: E402
from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence  # noqa: E402

DEFAULT_ROOT = r"D:\GitHub\theProjectSignbridge"
OUT_DIR = os.path.join(PROJECT_ROOT, "ml", "datasets", "gsl_sentences")
SPLIT_FILES = {"train": "gsl_split_SI_train.csv", "dev": "gsl_split_SI_dev.csv", "test": "gsl_split_SI_test.csv"}
SIGNER_RE = re.compile(r"signer(\d+)", re.IGNORECASE)
MANIFEST_FIELDS = ["file", "split", "signer", "scenario", "sample", "label", "frames_total", "hand_frames"]
META_FILE = "prep_meta.json"
DEFAULT_TRIM_MARGIN = 2  # ίδια προεπιλογή με Settings.sentence_margin στον server


def _load_helpers():
    """Επαναχρησιμοποιεί τις συναρτήσεις ανάγνωσης frames του prepare_open_dataset.py (ίδια συμπεριφορά)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_open_dataset.py")
    spec = importlib.util.spec_from_file_location("_prepare_open_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_helpers = _load_helpers()
iter_frames = _helpers.iter_frames
detect_hands = _helpers.detect_hands
_natural_key = _helpers._natural_key


def read_split(path: str) -> dict:
    """Γραμμές 'path|ΛΕΞΕΙΣ' -> {(φάκελος_signer, φάκελος_δείγμα): 'ΛΕΞΕΙΣ'}. Δέχεται και το πρόθεμα 'health1/'."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            raw, gloss = line.split("|", 1)
            parts = [p for p in raw.replace("\\", "/").split("/") if p]
            if len(parts) < 2:
                continue
            out[(parts[-2], parts[-1])] = " ".join(gloss.split())
    return out


def collect_samples(split_dir: str, scenarios) -> list:
    samples = []
    for split, fname in SPLIT_FILES.items():
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] Λείπει το {path}")
            continue
        for (parent, folder), label in read_split(path).items():
            if not any(parent.startswith(f"{s}_") for s in scenarios):
                continue
            m = SIGNER_RE.search(parent)
            samples.append({
                "split": split, "parent": parent, "folder": folder, "label": label,
                "signer": int(m.group(1)) if m else -1, "scenario": parent.split("_")[0],
            })
    return sorted(samples, key=lambda s: (s["parent"], _natural_key(s["folder"])))


def locate_sample(root: str, s: dict):
    for base in (os.path.join(root, s["scenario"]), root):
        p = os.path.join(base, s["parent"], s["folder"])
        if os.path.isdir(p):
            return p
    return None


def trim_sequence(sequence: list, margin):
    """
    Κρατά από το πρώτο ως το τελευταίο frame με χέρι, συν `margin` frames εκατέρωθεν.
    margin=None ή <0: χωρίς κόψιμο. Αν δεν υπάρχει χέρι πουθενά, επιστρέφεται ως έχει.
    Ίδιο αποτέλεσμα με το InferenceSession._finish_sentence του server (εκεί το περιθώριο είναι μηδενικά frames).
    """
    if margin is None or margin < 0 or not sequence:
        return sequence
    active = [i for i, k in enumerate(sequence) if k.any()]
    if not active:
        return sequence
    return sequence[max(0, active[0] - margin): active[-1] + margin + 1]


def extract_sequence(src: str, extractor, frames: int, trim_margin=DEFAULT_TRIM_MARGIN):
    """Επιστρέφει (πίνακας frames x 126, frames με χέρι, σύνολο frames ΠΡΙΝ το κόψιμο)."""
    sequence, with_hands = [], 0
    for frame in iter_frames(src):
        keypoints = extract_hand_keypoints(detect_hands(extractor, frame))
        if keypoints.any():
            with_hands += 1
        sequence.append(keypoints)
    total = len(sequence)
    sequence = trim_sequence(sequence, trim_margin)
    return normalize_sequence(sequence, target_frames=frames).astype(np.float32), with_hands, total


def _is_valid(path: str, frames: int) -> bool:
    """Υπάρχει ήδη και έχει το σωστό μήκος (αν άλλαξες το --frames, ξαναϋπολογίζεται)."""
    if not os.path.exists(path):
        return False
    try:
        return np.load(path, mmap_mode="r").shape[0] == frames
    except Exception:
        return False


def _row(s: dict, fname: str, total="", hands=""):
    return {"file": fname, "split": s["split"], "signer": s["signer"], "scenario": s["scenario"],
            "sample": f"{s['parent']}/{s['folder']}", "label": s["label"],
            "frames_total": total, "hand_frames": hands}


def _prepare_cache(out_dir: str, meta: dict) -> bool:
    """
    Ελέγχει αν τα ήδη υπολογισμένα landmarks έχουν φτιαχτεί με τις ίδιες ρυθμίσεις (frames, trim).
    Αν όχι, ο παλιός φάκελος μετονομάζεται (δεν σβήνεται) ώστε να μην αναμειχθούν δείγματα με διαφορετική
    προετοιμασία. Γράφει το prep_meta.json πριν ξεκινήσει η μετατροπή. Επιστρέφει True αν υπάρχει cache προς χρήση.
    """
    lm_dir = os.path.join(out_dir, "landmarks")
    meta_path = os.path.join(out_dir, META_FILE)
    os.makedirs(out_dir, exist_ok=True)

    old = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = None

    has_files = os.path.isdir(lm_dir) and any(n.endswith(".npy") for n in os.listdir(lm_dir))
    reuse = has_files and old == meta
    if has_files and not reuse:
        stale = os.path.join(out_dir, time.strftime("landmarks_stale_%Y%m%d_%H%M%S"))
        os.rename(lm_dir, stale)
        why = "δεν υπάρχει prep_meta.json (παλιότερη έκδοση χωρίς κόψιμο)" if old is None else f"άλλαξαν οι ρυθμίσεις ({old} -> {meta})"
        print(f"[INFO] Τα υπάρχοντα landmarks δεν ταιριάζουν ({why}).")
        print(f"       Μετονομάστηκαν σε {os.path.basename(stale)} και θα ξαναϋπολογιστούν (μπορείς να τα σβήσεις).")

    os.makedirs(lm_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return reuse


def convert(root=DEFAULT_ROOT, split_dir=None, scenarios=("health1",), frames=60, limit=None, min_hand_ratio=0.1,
            trim_margin=DEFAULT_TRIM_MARGIN):
    split_dir = split_dir or os.path.join(root, "GSL_split", "GSL_continuous")
    samples = collect_samples(split_dir, scenarios)
    if not samples:
        print(f"[ERROR] Δεν βρέθηκαν δείγματα για {', '.join(scenarios)} στα split αρχεία του {split_dir}")
        return None

    trim_margin = None if trim_margin is None or trim_margin < 0 else int(trim_margin)
    _prepare_cache(OUT_DIR, {"frames": frames, "trim_margin": trim_margin})
    lm_dir = os.path.join(OUT_DIR, "landmarks")
    print(f"[INFO] frames={frames}, κόψιμο: {'όχι' if trim_margin is None else f'ναι, περιθώριο {trim_margin} frames'}")
    print(f"[INFO] {len(samples)} δείγματα στα splits για: {', '.join(scenarios)}")

    rows, converted, skipped_low, missing = [], 0, 0, 0
    started = time.time()
    for s in samples:
        fname = f"{s['parent']}_{s['folder']}.npy"
        out_path = os.path.join(lm_dir, fname)
        if _is_valid(out_path, frames):
            rows.append(_row(s, fname))
            continue
        if limit is not None and converted >= limit:
            continue
        src = locate_sample(root, s)
        if src is None:
            missing += 1
            continue

        # Νέος extractor ανά δείγμα: το tracking του MediaPipe δεν «περνά» από πρόταση σε πρόταση.
        t0 = time.time()
        extractor = LandmarkExtractor()
        try:
            data, with_hands, total = extract_sequence(src, extractor, frames, trim_margin)
        finally:
            extractor.close()

        if total == 0 or with_hands / total < min_hand_ratio:
            print(f"  [SKIP] {s['parent']}/{s['folder']}: χέρια σε {with_hands}/{total} frames")
            skipped_low += 1
            continue

        np.save(out_path, data)
        converted += 1
        rows.append(_row(s, fname, total, with_hands))
        print(f"  [{converted}] {s['parent']}/{s['folder']}: {total} frames, χέρια σε {with_hands} "
              f"({time.time() - t0:.1f}s)  «{s['label']}»")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SUCCESS] {(time.time() - started) / 60:.1f} λεπτά | νέα: {converted} | χωρίς χέρια: {skipped_low} "
          f"| δεν βρέθηκε ο φάκελος: {missing}")
    for split in SPLIT_FILES:
        part = [r for r in rows if r["split"] == split]
        signers = sorted({r["signer"] for r in part})
        classes = len({r["label"] for r in part})
        print(f" -> {split:5s}: {len(part):4d} δείγματα, {classes:3d} προτάσεις, signers {signers}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Landmarks για αναγνώριση προτάσεων (CERTH GSL continuous)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="φάκελος που περιέχει το health1 κ.λπ. και το GSL_split")
    ap.add_argument("--split-dir", default=None, help="προεπιλογή: <root>\\GSL_split\\GSL_continuous")
    ap.add_argument("--scenarios", nargs="+", default=["health1"])
    ap.add_argument("--frames", type=int, default=60, help="μήκος αλληλουχίας μετά την ομοιόμορφη δειγματοληψία")
    ap.add_argument("--limit", type=int, default=None, help="μέγιστος αριθμός ΝΕΩΝ μετατροπών (δοκιμή ταχύτητας)")
    ap.add_argument("--min-hand-ratio", type=float, default=0.1)
    ap.add_argument("--trim-margin", type=int, default=DEFAULT_TRIM_MARGIN,
                    help="frames περιθώριο γύρω από το πρώτο/τελευταίο frame με χέρι (προεπιλογή 2)")
    ap.add_argument("--no-trim", action="store_true", help="κράτα όλο το βίντεο (χωρίς κόψιμο)")
    a = ap.parse_args()
    convert(a.root, a.split_dir, a.scenarios, a.frames, a.limit, a.min_hand_ratio,
            None if a.no_trim else a.trim_margin)
