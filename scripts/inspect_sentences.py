"""
Σύντομη διάγνωση των δεδομένων προτάσεων (ml/datasets/gsl_sentences). Δεν χρειάζεται MediaPipe ούτε το dataset.

Δείχνει:
  1. τις ρυθμίσεις με τις οποίες φτιάχτηκαν τα landmarks (prep_meta.json),
  2. ποιοι signers υπάρχουν σε κάθε split και πόσα δείγματα,
  3. πόσα δείγματα έχει κάθε label (πρόταση) σε train/dev/test, ώστε να φανούν παραλλαγές της ίδιας πρότασης,
  4. σε πόσα frames κάθε αρχείου φαίνεται χέρι (χαμηλό ποσοστό = προβληματική ανίχνευση).

Χρήση:
    python scripts\\inspect_sentences.py
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "ml", "datasets", "gsl_sentences")
SPLITS = ("train", "dev", "test")
LOW_HAND_RATIO = 0.5


def hand_ratio(path: str) -> float:
    """Ποσοστό frames του αρχείου όπου υπάρχει τουλάχιστον ένα χέρι (μη μηδενικά keypoints)."""
    arr = np.load(path)
    return float((np.abs(arr).sum(axis=1) > 0).mean()) if len(arr) else 0.0


def inspect(data_dir: str) -> int:
    manifest = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"[ERROR] Δεν βρέθηκε το {manifest}. Τρέξε πρώτα: python scripts\\prepare_sentences.py")
        return 1
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    meta_path = os.path.join(data_dir, "prep_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            print("Ρυθμίσεις προετοιμασίας (prep_meta.json):", json.load(f))
    else:
        print("[WARN] Λείπει το prep_meta.json: τα landmarks φτιάχτηκαν με παλιότερη έκδοση (χωρίς κόψιμο).")

    print("\n== Splits ==")
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        signers = sorted({int(r["signer"]) for r in part})
        print(f" {split:5s}: {len(part):4d} δείγματα, {len({r['label'] for r in part}):3d} labels, signers {signers}")
    signers_by_split = {s: {r["signer"] for r in rows if r["split"] == s} for s in SPLITS}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = sorted(signers_by_split[a] & signers_by_split[b], key=int)
        if shared:
            print(f" [!] Ίδιος signer σε {a} και {b}: {shared}")

    print("\n== Δείγματα ανά label (train / dev / test) ==")
    counts = defaultdict(Counter)
    for r in rows:
        counts[r["label"]][r["split"]] += 1
    for label in sorted(counts):
        c = counts[label]
        print(f" {c['train']:3d} /{c['dev']:3d} /{c['test']:3d}   {label}")
    train_counts = [c["train"] for c in counts.values() if c["train"]]
    if train_counts:
        print(f" labels: {len(counts)} | train ανά label: min {min(train_counts)}, max {max(train_counts)}")

    print("\n== Χέρια στα frames κάθε αρχείου ==")
    ratios, missing, low = [], 0, []
    for r in rows:
        path = os.path.join(data_dir, "landmarks", r["file"])
        if not os.path.exists(path):
            missing += 1
            continue
        ratio = hand_ratio(path)
        ratios.append((ratio, r))
        if ratio < LOW_HAND_RATIO:
            low.append((ratio, r))
    if missing:
        print(f" [WARN] {missing} αρχεία του manifest λείπουν από τον φάκελο landmarks.")
    if ratios:
        vals = np.array([v for v, _ in ratios])
        print(f" μέσος όρος {vals.mean():.0%} | διάμεσος {np.median(vals):.0%} | ελάχιστο {vals.min():.0%} "
              f"| 10ο εκατοστημόριο {np.percentile(vals, 10):.0%}")
        for split in SPLITS:
            part = np.array([v for v, r in ratios if r["split"] == split])
            if len(part):
                print(f"   {split:5s}: μέσος όρος {part.mean():.0%}")
        print(f" αρχεία με χέρι σε < {LOW_HAND_RATIO:.0%} των frames: {len(low)} από {len(ratios)}")
        for v, r in sorted(low, key=lambda t: t[0])[:10]:
            print(f"   {v:.0%}  {r['sample']}  (signer {r['signer']}, {r['split']})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Διάγνωση δεδομένων προτάσεων")
    ap.add_argument("--data-dir", default=DATA_DIR)
    raise SystemExit(inspect(ap.parse_args().data_dir))
