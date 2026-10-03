"""
Οργανώνει το CERTH GSL (isolated) dataset στη δομή που περιμένει το prepare_open_dataset.py:

    ml/raw_videos/gsl/<ΛΕΞΗ>/<κατηγορία>_signerN_repM_glosses_glossesNNNN/*.png

Το CSV έχει γραμμές της μορφής:
    police1_signer1_rep1_glosses/glosses0000|ΓΕΙΑ

ΣΗΜΑΝΤΙΚΟ: χρειάζονται οι φάκελοι "*_glosses" (μεμονωμένες λέξεις). Οι φάκελοι "*_sentences"
(προτάσεις) ΔΕΝ ανήκουν στο isolated CSV και αγνοούνται.

Χρήση:
    python scripts\\import_certh_dataset.py
    python scripts\\import_certh_dataset.py --videos-dir D:\\data\\certh --csv D:\\data\\train_greek_iso.csv
    python scripts\\import_certh_dataset.py --max-per-label 40
"""
import argparse
import os
import re
import shutil
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- ΠΡΟΣΑΡΜΟΣΕ ΑΥΤΑ ΤΑ PATHS (ή δώσε τα με --videos-dir / --csv) ---
# Φάκελος που περιέχει (σε οποιοδήποτε βάθος) τους φακέλους "*_glosses"
EXTRACTED_VIDEOS_DIR = r"D:\GitHub\theProjectSignbridge"
CSV_ANNOTATION_FILE = r"D:\GitHub\theProjectSignbridge\GSL_split\GSL_isolated\train_greek_iso.csv"
# ------------------------------------------------------------------

TARGET_RAW_DIR = os.path.join(PROJECT_ROOT, "ml", "raw_videos", "gsl")


def clean_label(raw: str) -> str:
    """Όνομα φακέλου από τη λέξη: αφαιρεί χαρακτήρες που δεν επιτρέπονται στα Windows."""
    label = re.sub(r'[<>:"/\\|?*\s]+', "_", str(raw).strip())
    return label.strip("._") or "UNKNOWN"


def build_index(root: str) -> dict:
    """
    Βρίσκει όλους τους φακέλους-δείγματα:  "police1_signer1_rep1_glosses/glosses0000" -> πλήρης διαδρομή.
    Δεν κατεβαίνει μέσα στους φακέλους με τα PNG (γρήγορο ακόμη και με δεκάδες χιλιάδες δείγματα)
    και προσπερνά τις προτάσεις (*_sentences).
    """
    index = {}
    for dirpath, dirnames, _ in os.walk(root):
        parent = os.path.basename(dirpath)
        if parent.endswith("_glosses"):
            for d in dirnames:
                index[f"{parent}/{d}"] = os.path.join(dirpath, d)
            dirnames[:] = []
        elif parent.endswith("_sentences"):
            dirnames[:] = []
    return index


def auto_organize_png_certh(videos_dir=EXTRACTED_VIDEOS_DIR, csv_file=CSV_ANNOTATION_FILE, max_per_label=None):
    if not os.path.exists(videos_dir):
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος: {videos_dir}")
        return
    if not os.path.exists(csv_file):
        print(f"[ERROR] Δεν βρέθηκε το CSV: {csv_file}")
        return

    print("--- Οργάνωση CERTH GSL (PNG sequences) ---")
    index = build_index(videos_dir)
    if not index:
        print(f"[ERROR] Δεν βρέθηκε κανένας φάκελος '*_glosses' μέσα στο {videos_dir}.")
        print("        Έχεις μόνο προτάσεις ('*_sentences'); κατέβασε τους φακέλους των μεμονωμένων λέξεων.")
        return

    folders = sorted({k.split("/")[0] for k in index})
    print(f"[INFO] Βρέθηκαν {len(index)} δείγματα σε {len(folders)} φακέλους: {', '.join(folders[:6])}"
          f"{' …' if len(folders) > 6 else ''}")

    # Το αρχείο είναι UTF-8 (το PowerShell απλώς το εμφανίζει λάθος)· utf-8-sig αγνοεί και το BOM.
    df = pd.read_csv(csv_file, sep="|", header=None, names=["path", "label"], encoding="utf-8-sig", dtype=str)

    copied = existing = 0
    missing = Counter()
    per_label = Counter()

    for raw_path, raw_label in zip(df["path"], df["label"], strict=True):
        parts = [p for p in str(raw_path).strip().replace("\\", "/").split("/") if p]
        if len(parts) < 2:
            missing["<άκυρη γραμμή>"] += 1
            continue
        parent, folder = parts[-2], os.path.splitext(parts[-1])[0]
        key = f"{parent}/{folder}"
        src = index.get(key)
        if src is None:
            missing[parent] += 1
            continue

        label = clean_label(raw_label)
        if max_per_label and per_label[label] >= max_per_label:
            continue

        # Το όνομα περιλαμβάνει κατηγορία/signer/επανάληψη, ώστε να μη συγκρούονται τα glosses0000 διαφορετικών signers.
        dst = os.path.join(TARGET_RAW_DIR, label, f"{parent}_{folder}")
        per_label[label] += 1
        if os.path.exists(dst):
            existing += 1
            continue
        shutil.copytree(src, dst)
        copied += 1

    print("\n[SUCCESS] Ολοκληρώθηκε!")
    print(f" -> Αντιγράφηκαν: {copied} δείγματα στο ml/raw_videos/gsl/ (ήδη υπήρχαν: {existing})")
    print(f" -> Διαφορετικές λέξεις: {len(per_label)}")
    if missing:
        total = sum(missing.values())
        print(f" -> Δεν βρέθηκαν {total} δείγματα του CSV (φυσιολογικό αν έχεις κατεβάσει μόνο μέρος του dataset).")
        print("    Οι πιο συχνοί φάκελοι που λείπουν:")
        for name, n in missing.most_common(5):
            print(f"      {name}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Οργάνωση CERTH GSL isolated dataset")
    ap.add_argument("--videos-dir", default=EXTRACTED_VIDEOS_DIR)
    ap.add_argument("--csv", default=CSV_ANNOTATION_FILE)
    ap.add_argument("--max-per-label", type=int, default=None,
                    help="μέγιστα δείγματα ανά λέξη (λιγότερο αντίγραφο / γρηγορότερη δοκιμή)")
    args = ap.parse_args()
    auto_organize_png_certh(args.videos_dir, args.csv, args.max_per_label)