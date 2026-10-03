"""
Κατεβάζει το hand_landmarker.task (MediaPipe Tasks) στο services/inference/.

    python scripts/download_models.py
"""
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
TARGET = os.path.join(PROJECT_ROOT, "services", "inference", "hand_landmarker.task")


def main() -> int:
    if os.path.isfile(TARGET) and os.path.getsize(TARGET) > 1_000_000:
        print(f"[SKIP] Υπάρχει ήδη: {TARGET}")
        return 0
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    tmp = TARGET + ".part"
    print(f"Κατέβασμα: {URL}")
    try:
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, TARGET)  # ατομική μετονομασία: δεν μένει μισό αρχείο αν κοπεί η λήψη
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[ERROR] Αποτυχία λήψης: {exc}")
        return 1
    print(f"[OK] Αποθηκεύτηκε: {TARGET} ({os.path.getsize(TARGET) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
