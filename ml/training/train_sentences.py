"""
Εκπαίδευση αναγνώρισης ΠΡΟΤΑΣΕΩΝ (CERTH GSL continuous) με το υπάρχον SignLanguageLSTM.

- Διαβάζει το ml/datasets/gsl_sentences/manifest.csv (φτιάχνεται από scripts/prepare_sentences.py).
- Εκπαιδεύει στο split "train", επιλέγει την καλύτερη εποχή με το "dev" και αναφέρει το τελικό
  αποτέλεσμα στο "test". Τα τρία σύνολα έχουν ΔΙΑΦΟΡΕΤΙΚΟΥΣ signers, άρα το test accuracy δείχνει
  πόσο καλά γενικεύει το μοντέλο σε άτομα που δεν έχει δει.
- Αποθηκεύει στο services/inference/weights/gsl_sentences_model.pth
  (ίδιο format με το train.py: model_state + labels, συν seq_len, mode="sentence", trim_margin και αποτελέσματα).
  Ο server διαβάζει από εκεί το μήκος αλληλουχίας και το ότι πρέπει να αποφασίζει στο τέλος της πρότασης.
  ΔΕΝ πειράζει το υπάρχον gsl_model.pth.

Χρήση:
    python -m ml.training.train_sentences
    python -m ml.training.train_sentences --epochs 120
"""
import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ml.models.sign_model import SignLanguageLSTM  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "ml", "datasets", "gsl_sentences")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "services", "inference", "weights", "gsl_sentences_model.pth")


def load_manifest(data_dir):
    path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Δεν βρέθηκε το {path}. Τρέξε πρώτα: python scripts\\prepare_sentences.py")
    rows = {"train": [], "dev": [], "test": []}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["split"], []).append(r)
    return rows


def load_meta(data_dir):
    """Ρυθμίσεις με τις οποίες φτιάχτηκαν τα landmarks (prepare_sentences.py). None αν λείπει."""
    path = os.path.join(data_dir, "prep_meta.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def to_arrays(rows, label_to_id, data_dir):
    X, y, dropped = [], [], 0
    for r in rows:
        if r["label"] not in label_to_id:
            dropped += 1  # πρόταση που δεν υπάρχει στο train: δεν μπορεί να αναγνωριστεί
            continue
        X.append(np.load(os.path.join(data_dir, "landmarks", r["file"])))
        y.append(label_to_id[r["label"]])
    if not X:
        return None, None, dropped
    return torch.from_numpy(np.stack(X).astype(np.float32)), torch.tensor(y, dtype=torch.long), dropped


def augment(x, shift=0.05, scale=0.1, noise=0.003):
    """
    Τυχαία μετατόπιση / κλίμακα / θόρυβος στις συντεταγμένες των χεριών (x: batch, T, 126).
    Προσομοιώνει άλλη θέση και απόσταση του signer από την κάμερα. Τα «χαμένα» χέρια (μηδενικά) μένουν μηδενικά.
    """
    b, t, _ = x.shape
    h = x.reshape(b, t, 2, 21, 3).clone()
    present = (h.abs().sum(dim=(3, 4)) > 0).unsqueeze(-1).unsqueeze(-1)  # (b, t, 2, 1, 1)
    s = torch.empty(b, 1, 1, 1).uniform_(1 - scale, 1 + scale)
    dx = torch.empty(b, 1, 1, 1).uniform_(-shift, shift)
    dy = torch.empty(b, 1, 1, 1).uniform_(-shift, shift)
    h[..., 0] = (h[..., 0] - 0.5) * s + 0.5 + dx
    h[..., 1] = (h[..., 1] - 0.5) * s + 0.5 + dy
    h[..., 2] = h[..., 2] * s
    h = h + torch.randn_like(h) * noise
    return (h * present).reshape(b, t, 126)


@torch.no_grad()
def evaluate(model, X, y, device, criterion):
    model.eval()
    logits = model(X.to(device))
    loss = criterion(logits, y.to(device)).item()
    preds = logits.argmax(dim=1).cpu()
    return loss, (preds == y).float().mean().item(), preds


def train(epochs=80, batch_size=16, lr=1e-3, weight_decay=1e-4, seed=0, data_dir=DATA_DIR, out_path=WEIGHTS_PATH):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = load_manifest(data_dir)
    if not rows["train"] or not rows["dev"] or not rows["test"]:
        raise RuntimeError("Το manifest χρειάζεται δείγματα και στα τρία splits (train/dev/test). "
                           f"Βρέθηκαν: { {k: len(v) for k, v in rows.items()} }")

    labels = sorted({r["label"] for r in rows["train"]})
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    Xtr, ytr, _ = to_arrays(rows["train"], label_to_id, data_dir)
    Xdev, ydev, d_dev = to_arrays(rows["dev"], label_to_id, data_dir)
    Xte, yte, d_te = to_arrays(rows["test"], label_to_id, data_dir)
    if Xdev is None or Xte is None:
        raise RuntimeError("Τα dev/test δεν έχουν καμία πρόταση που υπάρχει και στο train.")
    if d_dev or d_te:
        print(f"[WARN] Αγνοήθηκαν {d_dev} δείγματα dev και {d_te} test με προτάσεις που λείπουν από το train.")

    seq_len = Xtr.shape[1]
    meta = load_meta(data_dir)
    trim_margin = meta.get("trim_margin") if meta else None
    if trim_margin is None:
        print("[WARN] Τα landmarks δεν είναι «κομμένα» (ή λείπει το prep_meta.json). Ο server κόβει την πρόταση "
              "στα χέρια που εμφανίζονται, άρα η ακρίβεια live θα είναι χειρότερη. "
              "Ξανατρέξε: python scripts\\prepare_sentences.py")
    print(f"\n--- Εκπαίδευση προτάσεων: {len(labels)} κλάσεις, seq_len={seq_len} ---")
    for name, rr in (("train", rows["train"]), ("dev", rows["dev"]), ("test", rows["test"])):
        print(f" {name:5s}: {len(rr):4d} δείγματα, signers {sorted({int(r['signer']) for r in rr})}")
    counts = Counter(r["label"] for r in rows["train"])
    print(f" δείγματα train ανά πρόταση: min {min(counts.values())}, max {max(counts.values())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageLSTM(num_classes=len(labels), input_size=Xtr.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = augment(Xtr[idx]).to(device), ytr[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idx)

        dev_loss, dev_acc, _ = evaluate(model, Xdev, ydev, device, criterion)
        if dev_acc > best["acc"] or (dev_acc == best["acc"] and dev_loss < best["loss"]):
            best = {"acc": dev_acc, "loss": dev_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] loss {total / len(Xtr):.4f} | dev acc {dev_acc * 100:5.1f}%")

    model.load_state_dict(best["state"])
    _, test_acc, preds = evaluate(model, Xte, yte, device, criterion)
    print(f"\n[ΑΠΟΤΕΛΕΣΜΑ] καλύτερη εποχή {best['epoch']} | dev {best['acc'] * 100:.1f}% | "
          f"TEST (άγνωστοι signers) {test_acc * 100:.1f}%  (τυχαία επιλογή ≈ {100 / len(labels):.1f}%)")

    confusions = Counter((int(t), int(p)) for t, p in zip(yte, preds, strict=True) if t != p)
    if confusions:
        print(" Συχνότερες συγχύσεις στο test:")
        for (t, p), n in confusions.most_common(5):
            print(f"   {n}x  «{labels[t]}»  ->  «{labels[p]}»")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": labels, "seq_len": seq_len, "input_size": int(Xtr.shape[2]),
        "mode": "sentence", "trim_margin": trim_margin,
        "dev_acc": best["acc"], "test_acc": test_acc, "epoch": best["epoch"],
    }, out_path)
    with open(os.path.join(data_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"test_acc": test_acc, "dev_acc": best["acc"], "classes": len(labels), "seq_len": seq_len,
                   "trim_margin": trim_margin,
                   "confusions": [{"true": labels[t], "pred": labels[p], "count": n} for (t, p), n in confusions.items()]},
                  f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Αποθηκεύτηκε: {out_path}")
    return test_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Εκπαίδευση αναγνώρισης προτάσεων GSL")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.lr, seed=a.seed)
