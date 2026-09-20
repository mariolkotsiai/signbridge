import os
import sys
import numpy as np
import cv2

# Προσθήκη του root directory στο PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.inference.landmarks_pipeline import LandmarkExtractor
from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence


def process_video_file(video_path: str, extractor: LandmarkExtractor) -> np.ndarray:
    """
    Διαβάζει ένα αρχείο βίντεο και εξάγει κανονικοποιημένα landmarks 30 frames.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, results, _ = extractor.process_frame(frame)
        keypoints = extract_hand_keypoints(results)
        sequence_data.append(keypoints)

    cap.release()
    return normalize_sequence(sequence_data, target_frames=30)


def convert_dataset_videos(language='gsl'):
    """
    Διαβάζει όλα τα βίντεο από τον φάκελο ml/raw_videos/{language}
    και τα μετατρέπει σε .npy αρχεία στο ml/datasets/{language}/raw_landmarks.
    """
    raw_dir = os.path.join(PROJECT_ROOT, f"ml/raw_videos/{language}")
    out_dir = os.path.join(PROJECT_ROOT, f"ml/datasets/{language}/raw_landmarks")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος: {raw_dir}")
        return

    extractor = LandmarkExtractor()
    total_converted = 0

    # Περιήγηση σε κάθε φάκελο-ετικέτα (π.χ. ΓΕΙΑ, ΕΥΧΑΡΙΣΤΩ)
    for label in os.listdir(raw_dir):
        label_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(label_dir):
            continue

        video_files = [f for f in os.listdir(label_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"\n[INFO] Επεξεργασία {len(video_files)} βίντεο για τη λέξη: '{label}'")

        for idx, v_file in enumerate(video_files):
            v_path = os.path.join(label_dir, v_file)
            npy_data = process_video_file(v_path, extractor)

            save_name = f"{label}_open_{idx}_{int(np.random.randint(1000, 9999))}.npy"
            save_path = os.path.join(out_dir, save_name)
            np.save(save_path, npy_data)
            print(f"  [{idx+1}/{len(video_files)}] {v_file} -> {save_name}")
            total_converted += 1

    extractor.close()
    print(f"\n[SUCCESS] Ολοκληρώθηκε! Μετατράπηκαν συνολικά {total_converted} βίντεο.")


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else 'gsl'
    convert_dataset_videos(language=lang)