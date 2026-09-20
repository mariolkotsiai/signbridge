import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def convert_dataset_videos(input_folder: str, output_folder: str, label: str):
    """
    Μετατρέπει όλα τα βίντεο ενός φακέλου σε .npy αρχεία landmarks.
    """
    os.makedirs(output_folder, exist_ok=True)
    extractor = LandmarkExtractor()

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"[INFO] Βρέθηκαν {len(video_files)} βίντεο για τη λέξη: {label}")

    for idx, v_file in enumerate(video_files):
        v_path = os.path.join(input_folder, v_file)
        npy_data = process_video_file(v_path, extractor)

        save_name = f"{label}_open_{idx}_{int(np.random.randint(1000, 9999))}.npy"
        np.save(os.path.join(output_folder, save_name), npy_data)
        print(f"[{idx+1}/{len(video_files)}] Επεξεργάστηκε: {v_file} -> {save_name}")

    extractor.close()


if __name__ == "__main__":
    print("Έτοιμος μηχανισμός μετατροπής Open Datasets σε Landmarks!")