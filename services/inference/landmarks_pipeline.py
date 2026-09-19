
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Τυπικές 21 συνδέσεις σημείων χεριού (MediaPipe hand landmark topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # αντίχειρας
    (0, 5), (5, 6), (6, 7), (7, 8),          # δείκτης
    (5, 9), (9, 10), (10, 11), (11, 12),     # μέσος
    (9, 13), (13, 14), (14, 15), (15, 16),   # παράμεσος
    (13, 17), (17, 18), (18, 19), (19, 20),  # μικρό
    (0, 17),                                 # παλάμη
]


class LandmarkExtractor:
    """
    Εξαγωγέας σημείων (Landmarks) χεριών με το νέο MediaPipe Tasks API.
    """
    def __init__(self, model_path: str = MODEL_PATH, num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] Δεν βρέθηκε το μοντέλο: {model_path}\n"
                "Κατέβασέ το από: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray):
        """
        Επεξεργασία ενός frame, εξαγωγή landmarks, σχεδίαση και υπολογισμός latency.
        """
        start_time = time.perf_counter()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated_image = frame.copy()
        h, w = annotated_image.shape[:2]

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(annotated_image, points[start_idx], points[end_idx],
                             (0, 200, 0), 2)

                for (px, py) in points:
                    cv2.circle(annotated_image, (px, py), 4, (0, 0, 255), -1)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return annotated_image, result, latency_ms

    def close(self):
        self.landmarker.close()


def run_webcam_demo():
    """
    Εκτέλεση live capture από την webcam με FPS overlay.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Δεν ήταν δυνατή η πρόσβαση στην κάμερα (Index 0).")
        return

    try:
        extractor = LandmarkExtractor()
    except FileNotFoundError as e:
        print(e)
        cap.release()
        return

    prev_time = time.time()

    print("[INFO] Η κάμερα ξεκίνησε. Πίεσε 'q' για έξοδο.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Αποτυχία λήψης frame.")
            break

        annotated_frame, results, latency_ms = extractor.process_frame(frame)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            annotated_frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow("SignBridge - M1 Landmark Pipeline", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    run_webcam_demo()