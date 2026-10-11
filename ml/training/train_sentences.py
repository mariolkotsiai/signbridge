"""
Εκπαίδευση αναγνώρισης ΠΡΟΤΑΣΕΩΝ (CERTH GSL continuous) με το υπάρχον SignLanguageLSTM.

Τρεις τρόποι λειτουργίας (διαβάζουν το ml/datasets/gsl_sentences/manifest.csv του scripts/prepare_sentences.py):

1. Προεπιλογή (χωρίς flags): εκπαιδεύει στο split "train", επιλέγει την καλύτερη εποχή με το "dev" και αναφέρει
   το αποτέλεσμα στο "test". ΠΡΟΣΟΧΗ: αν τα dev και test έχουν τον ίδιο signer, η επιλογή εποχής «βλέπει»
   τον signer του test και το αποτέλεσμα είναι αισιόδοξο.

2. --loso (leave-one-signer-out): κάθε signer κρατιέται με τη σειρά έξω από την εκπαίδευση και το μοντέλο
   αξιολογείται ΜΟΝΟ σε αυτόν. Δεν γίνεται καμία επιλογή εποχής πάνω στον signer που ελέγχεται: κάθε fold
   εκπαιδεύεται για σταθερό αριθμό εποχών και κρατιέται το μοντέλο της τελευταίας. Η μέση ακρίβεια ανά
   signer (± τυπική απόκλιση) είναι η πιο ρεαλιστική εκτίμηση για «νέο άτομο» που διαθέτει το dataset.

3. --final: εκπαιδεύει ΕΝΑ μοντέλο σε ΟΛΟΥΣ τους signers (train + dev + test), για σταθερό αριθμό εποχών,
   και το γράφει ως services/inference/weights/gsl_sentences_model.pth (το προηγούμενο αντιγράφεται πρώτα
   σε gsl_sentences_model_prev_<ώρα>.pth). Μετά από αυτό το --play-test ΔΕΝ είναι ανεξάρτητη δοκιμή,
   γιατί το μοντέλο έχει δει και τους signers του test· για απόδοση δες την αναφορά του --loso.

Τα flags --loso και --final συνδυάζονται: το --final γράφει στο checkpoint τη συνολική ακρίβεια του --loso.

Χρήση:
    python -m ml.training.train_sentences
    python -m ml.training.train_sentences --loso
    python -m ml.training.train_sentences --loso --final --epochs 100
"""
import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
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
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------------- #
# Δεδομένα
# --------------------------------------------------------------------------- #
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


def all_rows(rows):
    """Όλα τα δείγματα του manifest (train + dev + test) σε μία λίστα."""
    return [r for part in rows.values() for r in part]


def signers_of(rows):
    return sorted({int(r["signer"]) for r in rows})


def split_by_signer(rows, held_out):
    """(δείγματα εκπαίδευσης, δείγματα του signer που κρατιέται έξω). Ποτέ ο ίδιος signer και στα δύο."""
    held = int(held_out)
    return ([r for r in rows if int(r["signer"]) != held],
            [r for r in rows if int(r["signer"]) == held])


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


@torch.no_grad()
def predict_proba(model, X, device):
    """Πιθανότητες softmax (N, κλάσεις) στη CPU."""
    model.eval()
    return torch.softmax(model(X.to(device)), dim=1).cpu()


# --------------------------------------------------------------------------- #
# Προεπιλεγμένη εκπαίδευση (train / dev / test)
# --------------------------------------------------------------------------- #
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
          f"TEST {test_acc * 100:.1f}%  (τυχαία επιλογή ≈ {100 / len(labels):.1f}%)")
    shared = sorted({int(r["signer"]) for r in rows["dev"]} & {int(r["signer"]) for r in rows["test"]})
    if shared:
        print(f"[ΠΡΟΣΟΧΗ] Ο signer {shared} είναι και στο dev και στο test: η εποχή επιλέχθηκε πάνω σε αυτόν, "
              f"άρα το TEST είναι αισιόδοξο. Για αξιόπιστη εκτίμηση τρέξε: python -m ml.training.train_sentences --loso")

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


# --------------------------------------------------------------------------- #
# Κοινός βρόχος εκπαίδευσης για --loso και --final (σταθερός αριθμός εποχών, χωρίς επιλογή εποχής)
# --------------------------------------------------------------------------- #
def _fit(Xtr, ytr, num_classes, epochs, batch_size, lr, weight_decay, seed, device, on_epoch=None):
    """Εκπαιδεύει νέο μοντέλο για ακριβώς `epochs` εποχές και επιστρέφει το μοντέλο της ΤΕΛΕΥΤΑΙΑΣ εποχής."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SignLanguageLSTM(num_classes=num_classes, input_size=Xtr.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        if on_epoch is not None:
            on_epoch(epoch, total / len(Xtr), model)
    model.eval()
    return model


def threshold_rows(conf, correct, thresholds=THRESHOLDS):
    """Για κάθε όριο: πόσα δείγματα θα εμφανίζονταν (εμπιστοσύνη >= όριο) και πόσα από αυτά είναι σωστά."""
    out = []
    for t in thresholds:
        shown = [bool(c) for p, c in zip(conf, correct, strict=True) if p >= t]
        out.append({"threshold": t, "shown": len(shown), "n": len(conf), "correct": sum(shown)})
    return out


def _load_pool(data_dir):
    """Όλα τα δείγματα όλων των splits ως tensors, με ενιαίο σύνολο κλάσεων (union)."""
    pool = all_rows(load_manifest(data_dir))
    if not pool:
        raise RuntimeError(f"Το manifest στο {data_dir} δεν έχει κανένα δείγμα.")
    labels = sorted({r["label"] for r in pool})
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    X, y, _ = to_arrays(pool, label_to_id, data_dir)
    signer = torch.tensor([int(r["signer"]) for r in pool], dtype=torch.long)
    return pool, labels, X, y, signer


# --------------------------------------------------------------------------- #
# Leave-one-signer-out
# --------------------------------------------------------------------------- #
def loso(epochs=80, batch_size=16, lr=1e-3, weight_decay=1e-4, seed=0, data_dir=DATA_DIR,
         report_path=None, curve_every=10):
    """
    Αξιολόγηση leave-one-signer-out πάνω σε ΟΛΟΥΣ τους signers του manifest. Κάθε fold εκπαιδεύεται για σταθερό
    αριθμό εποχών (καμία επιλογή εποχής με βάση τον signer που ελέγχεται). Προτάσεις που ο signer του fold είναι ο
    μόνος που τις έχει νοηματίσει δεν μπορούν να αναγνωριστούν (δεν υπάρχουν στην εκπαίδευση του fold): μετράνε ως
    λάθος στη συνολική ακρίβεια και αναφέρονται ξεχωριστά ως «μη προσβάσιμες».
    Επιστρέφει λεξικό με τα αποτελέσματα και γράφει το loso_report.json.
    """
    pool, labels, X, y, signer = _load_pool(data_dir)
    signers = signers_of(pool)
    if len(signers) < 2:
        raise RuntimeError(f"Το leave-one-signer-out χρειάζεται τουλάχιστον 2 signers, βρέθηκαν {signers}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Leave-one-signer-out: {len(signers)} signers {signers}, {len(pool)} δείγματα, "
          f"{len(labels)} προτάσεις, {epochs} εποχές ανά fold ---")

    folds, oof = [], []  # oof: προβλέψεις εκτός εκπαίδευσης, μία ανά δείγμα
    for s in signers:
        held = signer == s
        Xtr, ytr, Xte, yte = X[~held], y[~held], X[held], y[held]
        train_ids = set(int(v) for v in ytr)
        reachable = torch.tensor([int(v) in train_ids for v in yte])
        curve = {}

        def on_epoch(epoch, _loss, model, Xte=Xte, yte=yte, curve=curve):
            if curve_every and (epoch % curve_every == 0 or epoch == epochs):
                curve[epoch] = float((predict_proba(model, Xte, device).argmax(dim=1) == yte).float().mean())

        t0 = time.time()
        model = _fit(Xtr, ytr, len(labels), epochs, batch_size, lr, weight_decay, seed + s, device, on_epoch)
        conf, preds = predict_proba(model, Xte, device).max(dim=1)
        correct = preds == yte
        n_ok, n = int(correct.sum()), len(yte)
        n_reach = int(reachable.sum())
        reach_ok = int(correct[reachable].sum())
        fold = {"signer": s, "n_train": len(ytr), "n": n, "correct": n_ok, "acc": n_ok / n,
                "unreachable": n - n_reach, "reachable_n": n_reach, "reachable_correct": reach_ok,
                "reachable_acc": (reach_ok / n_reach) if n_reach else None,
                "curve": curve}
        folds.append(fold)
        for i in range(n):
            oof.append({"signer": s, "true": int(yte[i]), "pred": int(preds[i]), "conf": float(conf[i])})
        extra = f" (προσβάσιμες {reach_ok}/{n_reach} = {reach_ok / n_reach:.1%})" if n_reach != n else ""
        print(f" signer {s}: {n_ok}/{n} = {n_ok / n:.1%}{extra}  [{time.time() - t0:.0f} s]")

    accs = np.array([f["acc"] for f in folds])
    total_n = sum(f["n"] for f in folds)
    total_ok = sum(f["correct"] for f in folds)
    reach_n = sum(f["reachable_n"] for f in folds)
    reach_ok_all = sum(f["reachable_correct"] for f in folds)
    std = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
    pooled = total_ok / total_n
    reach_pooled = reach_ok_all / reach_n if reach_n else None

    per_label = {}
    for o in oof:
        d = per_label.setdefault(labels[o["true"]], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(o["true"] == o["pred"])
    confusions = Counter((o["true"], o["pred"]) for o in oof if o["true"] != o["pred"])
    thresholds = threshold_rows([o["conf"] for o in oof], [o["true"] == o["pred"] for o in oof])

    curve_mean = {}
    for e in sorted({e for f in folds for e in f["curve"]}):
        vals = [f["curve"][e] for f in folds if e in f["curve"]]
        curve_mean[e] = float(np.mean(vals))

    print(f"\n[LOSO] μέση ακρίβεια ανά signer {accs.mean():.1%} ± {std:.1%} "
          f"(ελάχιστο {accs.min():.1%}, μέγιστο {accs.max():.1%})")
    print(f"       συνολικά {total_ok}/{total_n} = {pooled:.1%}"
          + (f" | μόνο προσβάσιμες προτάσεις {reach_ok_all}/{reach_n} = {reach_pooled:.1%}" if reach_n != total_n else ""))
    print(f"       τυχαία επιλογή ≈ {100 / len(labels):.1f}%")
    unreachable_total = total_n - reach_n
    if unreachable_total:
        print(f" [!] {unreachable_total} δείγματα έχουν πρόταση που στο fold τους δεν υπάρχει στην εκπαίδευση "
              f"(παραλλαγές gloss με δείγματα από έναν μόνο signer)· αυτές δεν μπορούν να αναγνωριστούν.")

    if curve_mean:
        print(" Μέση ακρίβεια ανά εποχή (ενημερωτικά· ΔΕΝ επιλέγεται εποχή με βάση αυτήν): "
              + " | ".join(f"{e}: {a:.1%}" for e, a in curve_mean.items()))

    weakest = sorted(per_label.items(), key=lambda kv: (kv[1]["correct"] / kv[1]["n"], -kv[1]["n"]))[:6]
    print(" Πιο αδύναμες προτάσεις (εκτός εκπαίδευσης):")
    for lab, d in weakest:
        print(f"   {d['correct']:2d}/{d['n']:<2d} = {d['correct'] / d['n']:4.0%}   «{lab}»")
    if confusions:
        print(" Συχνότερες συγχύσεις:")
        for (t, p), n in confusions.most_common(6):
            print(f"   {n}x  «{labels[t]}»  ->  «{labels[p]}»")

    print("\n Όριο  | εμφανίζονται | σωστές ανάμεσά τους  (μόνο μοντέλο, χωρίς τον server)")
    for r in thresholds:
        prec = f"{r['correct']}/{r['shown']} = {r['correct'] / r['shown']:.0%}" if r["shown"] else "-"
        print(f" {r['threshold']:4.0%}  | {r['shown']:3d}/{r['n']:<3d}     | {prec}")

    result = {
        "epochs": epochs, "seed": seed, "signers": signers, "classes": len(labels), "samples": total_n,
        "mean_acc": float(accs.mean()), "std_acc": std, "min_acc": float(accs.min()), "max_acc": float(accs.max()),
        "pooled_acc": pooled, "pooled_reachable_acc": reach_pooled, "unreachable": unreachable_total,
        "folds": folds, "per_label": per_label, "thresholds": thresholds, "curve_mean": curve_mean,
        "confusions": [{"true": labels[t], "pred": labels[p], "count": n} for (t, p), n in confusions.items()],
    }
    report_path = report_path or os.path.join(data_dir, "loso_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[SUCCESS] Αναφορά: {report_path}")
    return result


# --------------------------------------------------------------------------- #
# Τελικό μοντέλο σε όλους τους signers
# --------------------------------------------------------------------------- #
def backup_existing(path):
    """Αντιγράφει ένα υπάρχον checkpoint σε <όνομα>_prev_<ώρα>.pth (δεν διαβάζεται από τον server). None αν δεν υπάρχει."""
    if not os.path.exists(path):
        return None
    base, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest, n = f"{base}_prev_{stamp}{ext}", 1
    while os.path.exists(dest):
        dest = f"{base}_prev_{stamp}_{n}{ext}"
        n += 1
    shutil.copy2(path, dest)
    return dest


def train_final(epochs=80, batch_size=16, lr=1e-3, weight_decay=1e-4, seed=0, data_dir=DATA_DIR,
                out_path=WEIGHTS_PATH, loso_acc=None):
    """
    Εκπαιδεύει ένα μοντέλο σε ΟΛΟΥΣ τους signers (train + dev + test) για σταθερό αριθμό εποχών και το γράφει στο
    out_path στο ίδιο format με την train(). Το προηγούμενο checkpoint αντιγράφεται πρώτα. Επιστρέφει το out_path.
    """
    pool, labels, X, y, signer = _load_pool(data_dir)
    signers = signers_of(pool)
    seq_len = int(X.shape[1])
    meta = load_meta(data_dir)
    trim_margin = meta.get("trim_margin") if meta else None
    if trim_margin is None:
        print("[WARN] Τα landmarks δεν είναι «κομμένα» (ή λείπει το prep_meta.json): ο server κόβει την πρόταση "
              "στα χέρια που εμφανίζονται, άρα η ακρίβεια live θα είναι χειρότερη. "
              "Ξανατρέξε: python scripts\\prepare_sentences.py")
    print(f"\n--- Τελικό μοντέλο: {len(pool)} δείγματα, signers {signers}, {len(labels)} προτάσεις, "
          f"seq_len={seq_len}, {epochs} εποχές ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model = _fit(X, y, len(labels), epochs, batch_size, lr, weight_decay, seed, device)
    train_acc = float((predict_proba(model, X, device).argmax(dim=1) == y).float().mean())
    print(f" ακρίβεια στα δεδομένα εκπαίδευσης {train_acc:.1%} (δεν λέει τίποτα για νέους signers)  "
          f"[{time.time() - t0:.0f} s]")

    backup = backup_existing(out_path)
    if backup:
        print(f" Το προηγούμενο μοντέλο αντιγράφηκε στο: {backup}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": labels, "seq_len": seq_len, "input_size": int(X.shape[2]),
        "mode": "sentence", "trim_margin": trim_margin,
        "trained_on": "all_signers", "signers": signers, "epochs": int(epochs),
        "loso_acc": None if loso_acc is None else float(loso_acc),
    }, out_path)
    print(f"[SUCCESS] Αποθηκεύτηκε: {out_path}")
    print("[ΠΡΟΣΟΧΗ] Το μοντέλο έχει δει ΟΛΟΥΣ τους signers, άρα το --play-test δεν είναι πια ανεξάρτητη δοκιμή. "
          "Η αναμενόμενη ακρίβεια σε νέο άτομο είναι αυτή της αναφοράς --loso.")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Εκπαίδευση αναγνώρισης προτάσεων GSL")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loso", action="store_true",
                    help="αξιολόγηση leave-one-signer-out (σταθερές εποχές, χωρίς επιλογή εποχής στον signer του test)")
    ap.add_argument("--final", action="store_true",
                    help="εκπαίδευση ενός μοντέλου σε ΟΛΟΥΣ τους signers και αντικατάσταση του gsl_sentences_model.pth")
    ap.add_argument("--data-dir", default=DATA_DIR, help="φάκελος με manifest.csv και landmarks/")
    ap.add_argument("--out", default=WEIGHTS_PATH, help="πού γράφεται το checkpoint (προεπιλογή: φάκελος weights του server)")
    ap.add_argument("--report", default=None, help="διαδρομή του loso_report.json (προεπιλογή: μέσα στο --data-dir)")
    a = ap.parse_args(argv)

    if not a.loso and not a.final:
        train(a.epochs, a.batch_size, a.lr, seed=a.seed, data_dir=a.data_dir, out_path=a.out)
        return 0

    loso_acc = None
    if a.loso:
        res = loso(a.epochs, a.batch_size, a.lr, seed=a.seed, data_dir=a.data_dir, report_path=a.report)
        loso_acc = res["pooled_acc"]
    if a.final:
        train_final(a.epochs, a.batch_size, a.lr, seed=a.seed, data_dir=a.data_dir, out_path=a.out,
                    loso_acc=loso_acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
