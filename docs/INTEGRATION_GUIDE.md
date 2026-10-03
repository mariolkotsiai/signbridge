# SignBridge – Οδηγός ενσωμάτωσης (backend + frontend)

Αυτός ο οδηγός συνδέει τα κομμάτια που ήδη έχεις (LSTM, `normalize.py`, `LandmarkExtractor`, scripts δεδομένων)
με ένα FastAPI backend και ένα React/TypeScript UI, ώστε η ροή του διαγράμματος να δουλεύει τοπικά:

```
Webcam → React (JPEG, 15 fps) → WebSocket → FastAPI → MediaPipe → buffer 30 frames
       → normalize_sequence → LSTM (CPU) → threshold + ψηφοφορία → JSON → υπότιτλοι
```

## 1. Τι προστέθηκε / άλλαξε

| Αρχείο | Ενέργεια | Ρόλος |
|---|---|---|
| `services/inference/live_inference.py` | **αντικατάσταση** | Engine: φόρτωση `.pth`, sliding buffer, inference, ψηφοφορία. Δεν εξαρτάται από FastAPI. Έχει και CLI demo. |
| `services/api/main.py`, `schemas.py` | νέο | FastAPI gateway: `WS /ws/infer`, `GET /api/health`, `POST /api/models/reload`, `POST /api/predict/sequence` |
| `services/config.py` | νέο | Όλες οι ρυθμίσεις, με override από μεταβλητές `SIGNBRIDGE_*` |
| `services/inference/landmarks_pipeline.py` | **μικρή αλλαγή** | Προστέθηκε μέθοδος `detect()` (χωρίς σχεδίαση, με αυστηρά αύξοντα timestamps). Το `process_frame` δεν πειράχτηκε. |
| `scripts/download_models.py` | νέο | Κατεβάζει το `hand_landmarker.task` |
| `requirements.txt`, `.env.example` | ενημέρωση | Προστέθηκαν torch, scikit-learn, fastapi, uvicorn, httpx, yt-dlp |
| `ml/preprocessing/__init__.py`, `ml/training/__init__.py` | νέα (κενά) | Επιτρέπουν `python -m ml.training.train` |
| `frontend/` | νέο | Vite + React + TypeScript |
| `tests/`, `pytest.ini` | νέο | 34 tests |

> ⚠️ **Πριν αντικαταστήσεις το `live_inference.py`:** το αρχείο που ανέβασες εμφανίστηκε ως «binary» (συνήθως σημαίνει
> UTF-16, π.χ. από PowerShell `>`). Δεν μπόρεσα να το διαβάσω, άρα **δεν ξέρω τι περιείχε**. Κράτα αντίγραφο
> (`copy live_inference.py live_inference.py.bak`) αν έχει κάτι που θες να διατηρήσεις.

## 2. Εγκατάσταση (Windows / PowerShell, από το root του repo)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CPU PyTorch πρώτα (μικρό κατέβασμα, αρκεί για το ADR 001)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Μοντέλο χεριών του MediaPipe -> services/inference/hand_landmarker.task
python scripts\download_models.py

# Frontend
cd frontend
npm install
cd ..
```

## 3. Βάρη του μοντέλου (`.pth`)

Ο server διαβάζει `services/inference/weights/<γλώσσα>_model.pth` – ακριβώς εκεί που γράφει το `train.py`:
`{'model_state': ..., 'labels': [...]}`. Δεν χρειάζεται μετατροπή. Αν δεν έχεις εκπαιδεύσει ακόμα:

```powershell
python scripts\download_open_samples.py          # δείγματα βίντεο
python scripts\prepare_open_dataset.py gsl       # -> ml\datasets\gsl\raw_landmarks\*.npy
python scripts\prepare_open_dataset.py asl
python -m ml.training.train                      # -> services\inference\weights\{gsl,asl}_model.pth
```

Μετά από νέα εκπαίδευση **δεν χρειάζεται restart**: το επόμενο session ανιχνεύει το νέο αρχείο, ή κάλεσε
`POST http://127.0.0.1:8000/api/models/reload`.
Η αρχιτεκτονική (hidden size, layers) διαβάζεται από τα ίδια τα βάρη, άρα αλλαγές στο `SignLanguageLSTM` δεν σπάνε τη φόρτωση.

## 4. Εκτέλεση

**Ανάπτυξη (2 terminals):**

```powershell
# Terminal 1 – backend (root του repo)
uvicorn services.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 – frontend
cd frontend
npm run dev          # http://localhost:5173
```

Ο Vite προωθεί αυτόματα τα `/api` και `/ws` στο `127.0.0.1:8000` (`frontend/vite.config.ts`), οπότε δεν υπάρχει CORS.

**«Ένα process» (μετά από `npm run build`):** το FastAPI σερβίρει το `frontend/dist` στο `http://127.0.0.1:8000/`.

**Χωρίς browser** (έλεγχος μοντέλου + κάμερας): `python -m services.inference.live_inference --language gsl --show`
(`--source video.mp4` για αρχείο βίντεο).

**Γρήγορος έλεγχος του μοντέλου χωρίς κάμερα:**

```powershell
curl http://127.0.0.1:8000/api/health
```

## 5. Πρωτόκολλο WebSocket (για αλλαγές στον client)

`ws://host/ws/infer?language=gsl|asl`

| Κατεύθυνση | Μήνυμα |
|---|---|
| server → client | `{"type":"ready","labels":[…],"window_frames":30,"threshold":0.7,…}` |
| client → server | **binary** JPEG (ένα τη φορά· ο client περιμένει απάντηση πριν στείλει το επόμενο) |
| server → client | `{"type":"result","state":"idle\|collecting\|sign\|uncertain","hands":[[[x,y]×21]],"prediction":{label,confidence,accepted,top_k,stable_label,committed}\|null,"buffer":{filled,size},"latency_ms","fps"}` |
| client → server | `{"type":"config","threshold":0.8}` · `{"type":"reset"}` · `{"type":"ping"}` |
| server → client | `{"type":"frame_error",…}` (μη μοιραίο) ή `{"type":"error",…}` + κλείσιμο με κωδικό 4400/4404/4429/4500/4501 |

Ένας **νέος υπότιτλος** εμφανίζεται όταν `prediction.committed === true` (μία φορά ανά νόημα).

## 6. Πώς αποφασίζεται ένας υπότιτλος

1. Κάθε frame → `extract_hand_keypoints` (το **ίδιο** με το training → 126 τιμές).
2. Buffer έως 30 frames. Το inference τρέχει κάθε `STRIDE`=3 frames, αφού υπάρχουν ≥ `MIN_FRAMES`=15 και ≥ 40% των frames έχουν χέρι.
3. Ο buffer περνά από `normalize_sequence` → (30, 126), όπως στο training, και μετά από το LSTM + softmax.
4. `confidence ≥ threshold` → ψήφος για το label, αλλιώς ψήφος «άγνωστο» (`uncertain`).
5. Χρειάζονται `MIN_VOTES`=3 από τις τελευταίες 5 ίδιες ψήφους → «σταθερό» label → **ένας** υπότιτλος.
6. Αν δεν φαίνονται χέρια για `IDLE_RESET_FRAMES`=8 frames → `idle`: καθαρίζει ο buffer και επιτρέπεται να ξαναεμφανιστεί η ίδια λέξη.

Το μοντέλο δεν έχει κλάση «κανένα νόημα», οπότε ο έλεγχος παρουσίας χεριών και το threshold είναι αυτά που αποτρέπουν τις τυχαίες προβλέψεις.

## 7. Ρυθμίσεις που αξίζει να δοκιμάσεις

Στο `.env` (ή από το slider στο UI για το threshold):

| Μεταβλητή | Πότε την αλλάζεις |
|---|---|
| `SIGNBRIDGE_THRESHOLD` | Με μικρό dataset οι πιθανότητες είναι χαμηλές → χαμήλωσέ το (0.5–0.6) για δοκιμές |
| `SIGNBRIDGE_WINDOW_FRAMES` | Τα νοήματα του dataset σου διαρκούν πολύ περισσότερο/λιγότερο από ~2 s (στα 15 fps) |
| `SIGNBRIDGE_MIN_VOTES` / `VOTE_WINDOW` | Τρεμοπαίζουν οι υπότιτλοι → αύξησε· αργούν → μείωσε |
| `SIGNBRIDGE_MAX_SESSIONS` | Πόσες ταυτόχρονες κάμερες δέχεται (κάθε μία έχει δικό της MediaPipe) |

Στο `frontend/src/config.ts`: `STREAM.fps` (15), `jpegQuality`, `maxWidth`.

## 8. Γνωστοί περιορισμοί (διάβασέ τους πριν κρίνεις την ακρίβεια)

1. **Dataset:** το `download_open_samples.py` φέρνει 1–2 βίντεο ανά λέξη. Το LSTM με τόσα δείγματα απομνημονεύει, δεν γενικεύει· το «Val Accuracy» με 4 δείγματα δεν σημαίνει κάτι. Για πραγματική αναγνώριση χρειάζονται δεκάδες δείγματα ανά νόημα, από διαφορετικούς signers.
2. **Σειρά χεριών:** το `extract_hand_keypoints` βάζει το «1ο χέρι που ανιχνεύτηκε» στη θέση *left* και το 2ο στη *right*, χωρίς να κοιτά το handedness του MediaPipe. Η σειρά μπορεί να αλλάζει μεταξύ frames. Το έκανα συμβατό με όσα έχεις εκπαιδεύσει, αλλά είναι πιθανή πηγή θορύβου· αν αλλάξεις τον κανόνα πρέπει να ξαναεκπαιδεύσεις.
3. **Ακατέργαστες συντεταγμένες:** τα features είναι απόλυτες θέσεις στην εικόνα (όχι σχετικές με τον καρπό/ώμους, χωρίς κλίμακα). Αλλάζει η απόσταση ή η θέση του signer ⇒ αλλάζουν τα features. Ενδεχόμενη βελτίωση για το επόμενο training.
4. **`scripts/extract_landmarks.py`:** χρησιμοποιεί `mp.solutions.holistic` και παράγει 225 features – **ασύμβατο** με το live pipeline (126) και με τα νεότερα mediapipe που δεν έχουν πια το `mp.solutions`. Ο server θα το απορρίψει με σαφές μήνυμα αν εκπαιδεύσεις από εκεί.
5. **Αυτό που ΔΕΝ δοκίμασα:** πραγματική κάμερα, πραγματικό MediaPipe με πραγματικά χέρια (δεν είχα πρόσβαση στο `hand_landmarker.task`) και το React UI μέσα σε browser. Δοκιμάστηκαν: engine, API, πραγματικό uvicorn + WebSocket, loading βαρών από το δικό σου `train.py`, `tsc` και `vite build`.
6. **Ασφάλεια:** το API δεν έχει authentication· τρέξ’ το με `--host 127.0.0.1` (όχι `0.0.0.0`) εκτός αν το χρειάζεσαι και προσθέσεις προστασία.

## 9. Αντιμετώπιση προβλημάτων

| Σύμπτωμα | Αιτία / λύση |
|---|---|
| Κόκκινο μήνυμα «Λείπει το hand_landmarker.task» | `python scripts\download_models.py` |
| Το κουμπί γλώσσας γράφει «χωρίς μοντέλο» | Λείπει το `<γλώσσα>_model.pth` → εκπαίδευση (βλ. §3). Η αιτία φαίνεται στο tooltip |
| «Δεν βρέθηκε το backend» | Δεν τρέχει το uvicorn ή τρέχει σε άλλη θύρα → όρισε `VITE_BACKEND_URL` στο `frontend/.env` |
| Κάμερα: «απαιτεί ασφαλές πλαίσιο» | Άνοιξε το UI από `http://localhost…` (όχι από IP του δικτύου) ή με https |
| Συνεχώς `uncertain` | Χαμήλωσε το slider· ή το μοντέλο έχει εκπαιδευτεί με πολύ λίγα δεδομένα (§8.1) |
| Κλείσιμο WebSocket με 4429 | Ξεπεράστηκε το `MAX_SESSIONS` (π.χ. δύο ανοιχτά tabs) |
| `ModelLoadError … input_size=225` | Εκπαίδευση από `extract_landmarks.py` – βλ. §8.4 |
| Το FPS είναι χαμηλό | Δες `latency_ms` στο panel. Αν > 60 ms: μείωσε `STREAM.maxWidth` ή `STREAM.fps` |

## 10. Tests

```powershell
pip install -r requirements.txt
python -m pytest -q
```

Το test `tests/conftest.py` εκπαιδεύει πραγματικά με το `ml/training/train.py` σε συνθετικά δεδομένα και φορτώνει το αποτέλεσμα στον server, ώστε να πιάνει ασυμβατότητες στο format των βαρών. Δεν απαιτεί κάμερα ή `hand_landmarker.task`.
