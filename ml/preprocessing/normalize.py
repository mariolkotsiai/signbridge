import numpy as np

def extract_hand_keypoints(results) -> np.ndarray:
    """
    Εξάγει τις συντεταγμένες (X, Y, Z) για αριστερό και δεξί χέρι
    από το MediaPipe Tasks HandLandmarkerResult.
    """
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if hasattr(results, 'hand_landmarks') and results.hand_landmarks:
        # 1ο Χέρι
        if len(results.hand_landmarks) > 0:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]).flatten()
        # 2ο Χέρι
        if len(results.hand_landmarks) > 1:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[1]]).flatten()

    return np.concatenate([lh, rh])


def normalize_sequence(sequence: np.ndarray, target_frames: int = 30) -> np.ndarray:
    """
    Κανονικοποιεί το μήκος της αλληλουχίας frames σε σταθερό target_frames (30).
    """
    num_frames = len(sequence)
    if num_frames == 0:
        return np.zeros((target_frames, 126))

    if num_frames == target_frames:
        return np.array(sequence)

    indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
    return np.array(sequence)[indices]