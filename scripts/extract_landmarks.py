import os
import numpy as np
import cv2
import mediapipe as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "ml", "raw_videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "processed_landmarks")

mp_holistic = mp.solutions.holistic

POSE_POINTS = 33
HAND_POINTS = 21


def extract_landmarks_from_path(input_path, holistic):
    """
    Διαβάζει είτε βίντεο (.mp4) είτε φάκελο με εικόνες (.png/.jpg)
    και επιστρέφει numpy array σχήματος (num_frames, (33 + 21 + 21) * 3).
    """
    sequence = []

    # ΠΕΡΙΠΤΩΣΗ 1: Φάκελος με PNG/JPG Frames (CERTH Dataset)
    if os.path.isdir(input_path):
        image_files = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            pose = (np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
                    if results.pose_landmarks else np.zeros(POSE_POINTS * 3))
            left_hand = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                         if results.left_hand_landmarks else np.zeros(HAND_POINTS * 3))
            right_hand = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                          if results.right_hand_landmarks else np.zeros(HAND_POINTS * 3))

            sequence.append(np.concatenate([pose, left_hand, right_hand]))

    # ΠΕΡΙΠΤΩΣΗ 2: Αρχείο Βίντεο MP4
    elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            pose = (np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
                    if results.pose_landmarks else np.zeros(POSE_POINTS * 3))
            left_hand = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                         if results.left_hand_landmarks else np.zeros(HAND_POINTS * 3))
            right_hand = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                          if results.right_hand_landmarks else np.zeros(HAND_POINTS * 3))

            sequence.append(np.concatenate([pose, left_hand, right_hand]))
        cap.release()

    return np.array(sequence, dtype=np.float32)


def process_all_raw_data():
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος {RAW_VIDEOS_DIR}.")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for lang in sorted(os.listdir(RAW_VIDEOS_DIR)):
            lang_dir = os.path.join(RAW_VIDEOS_DIR, lang)
            if not os.path.isdir(lang_dir):
                continue

            for label in sorted(os.listdir(lang_dir)):
                label_dir = os.path.join(lang_dir, label)
                if not os.path.isdir(label_dir):
                    continue

                out_dir = os.path.join(OUTPUT_DIR, lang, label)
                os.makedirs(out_dir, exist_ok=True)

                for item in sorted(os.listdir(label_dir)):
                    item_path = os.path.join(label_dir, item)
                    sample_id = os.path.splitext(item)[0]
                    out_path = os.path.join(out_dir, f"{sample_id}.npy")

                    if os.path.exists(out_path):
                        print(f"  [SKIP] Υπάρχει ήδη: {out_path}")
                        continue

                    print(f"Εξαγωγή landmarks: [{lang.upper()}] {label} / {item}...")
                    try:
                        sequence = extract_landmarks_from_path(item_path, holistic)
                        if sequence.size == 0:
                            print(f"  [WARNING] Κενή ακολουθία: {item_path}")
                            continue
                        np.save(out_path, sequence)
                        print(f"  [OK] shape={sequence.shape} -> {out_path}")
                    except Exception as e:
                        print(f"  [ERROR] Αποτυχία εξαγωγής: {e}")


if __name__ == "__main__":
    process_all_raw_data()
    print("\n[SUCCESS] Η εξαγωγή landmarks ολοκληρώθηκε!")