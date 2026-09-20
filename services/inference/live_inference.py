import os
import sys
import time
import collections
import cv2
import numpy as np
import torch

# Δυναμική προσθήκη του root directory στο PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.inference.landmarks_pipeline import LandmarkExtractor
from ml.preprocessing.normalize import extract_hand_keypoints
from ml.models.sign_model import SignLanguageLSTM


class RealTimeSignTranslator:
    """
    Μηχανή Real-Time Μετάφρασης με Sliding Window.
    """
    def __init__(self, language='gsl', confidence_threshold=0.20):
        self.language = language
        self.threshold = confidence_threshold
        
        # Φόρτωση Βαρών & Labels
        weights_path = os.path.join(PROJECT_ROOT, f"services/inference/weights/{language}_model.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[ERROR] Δεν βρέθηκαν τα βάρη: {weights_path}. Τρέξε πρώτα το train.py!")

        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        self.labels = checkpoint['labels']
        
        # Δημιουργία μοντέλου με τη σωστή διάσταση εξόδου
        self.model = SignLanguageLSTM(num_classes=len(self.labels))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        # Sliding Window Buffer (30 frames)
        self.frame_buffer = collections.deque(maxlen=30)

    def predict_frame(self, keypoints: np.ndarray):
        """
        Προσθέτει τα νέα keypoints στο buffer και εκτελεί inference όταν το buffer είναι πλήρες.
        """
        self.frame_buffer.append(keypoints)

        if len(self.frame_buffer) < 30:
            return f"Syllogi ({len(self.frame_buffer)}/30)...", 0.0

        # Προετοιμασία Tensor (1, 30, 126)
        input_tensor = torch.tensor(np.array(self.frame_buffer), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probabilities, dim=1)
            
            prob_val = conf.item()
            predicted_label = self.labels[pred_idx.item()]

            if prob_val >= self.threshold:
                return predicted_label, prob_val

        return "Low Confidence", prob_val


def run_live_translation(language='gsl'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Δεν ήταν δυνατή η πρόσβαση στην κάμερα.")
        return

    try:
        extractor = LandmarkExtractor()
        translator = RealTimeSignTranslator(language=language, confidence_threshold=0.20)
    except Exception as e:
        print(f"[ERROR] {e}")
        cap.release()
        return

    print(f"\n--- Live Μετάφραση [{language.upper()}] Ξεκίνησε ---")
    print("Πίεσε 'q' για έξοδο.\n")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Εξαγωγή Landmarks
        annotated_frame, results, latency_ms = extractor.process_frame(frame)
        keypoints = extract_hand_keypoints(results)

        # 2. Live Inference
        label, confidence = translator.predict_frame(keypoints)

        # 3. Υπολογισμός FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # 4. Live UI Overlay
        cv2.rectangle(annotated_frame, (0, 0), (640, 75), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame, f"TRANSLATION: {label} ({confidence*100:.1f}%)",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.putText(
            annotated_frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Lang: {language.upper()}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        cv2.imshow("SignBridge - M3 Real-Time Live Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    lang_arg = sys.argv[1] if len(sys.argv) > 1 else 'gsl'
    run_live_translation(language=lang_arg)