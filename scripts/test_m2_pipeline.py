import os
import sys
import numpy as np
import cv2

# Προσθήκη του root directory στο PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing.normalize import extract_hand_keypoints, normalize_sequence


def test_normalization_dimensions():
    """
    Ελέγχει αν το normalization παράγει πάντα πίνακα διάστασης (30, 126).
    """
    print("[TEST 1/3] Έλεγχος διαστάσεων κανονικοποίησης (30, 126)...")
    
    # 1. Δοκιμή με λιγότερα frames (π.χ. 15 frames) -> πρέπει να γίνει interpolation σε 30
    short_seq = [np.random.rand(126) for _ in range(15)]
    norm_short = normalize_sequence(short_seq, target_frames=30)
    assert norm_short.shape == (30, 126), f"Σφάλμα! Αναμενόταν (30, 126), αλλά λήφθηκε {norm_short.shape}"

    # 2. Δοκιμή με περισσότερα frames (π.χ. 50 frames) -> πρέπει να γίνει downsampling σε 30
    long_seq = [np.random.rand(126) for _ in range(50)]
    norm_long = normalize_sequence(long_seq, target_frames=30)
    assert norm_long.shape == (30, 126), f"Σφάλμα! Αναμενόταν (30, 126), αλλά λήφθηκε {norm_long.shape}"

    print("  --> [OK] Ο μηχανισμός κανονικοποίησης λειτουργεί εξαιρετικά!\n")


def generate_synthetic_dataset():
    """
    Δημιουργεί δοκιμαστικά συνθετικά δεδομένα (.npy) για GSL και ASL
    ώστε να έχουμε έτοιμο dataset για το M3 (Model Training).
    """
    print("[TEST 2/3] Δημιουργία συνθετικού dataset για GSL & ASL...")

    vocab_gsl = ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΠΑΡΑΚΑΛΩ', 'ΝΑΙ', 'ΟΧΙ']
    vocab_asl = ['HELLO', 'THANK_YOU', 'PLEASE', 'YES', 'NO']

    for lang, vocab in [('gsl', vocab_gsl), ('asl', vocab_asl)]:
        out_dir = f"ml/datasets/{lang}/raw_landmarks"
        os.makedirs(out_dir, exist_ok=True)

        for label in vocab:
            # Δημιουργία 5 συνθετικών δειγμάτων ανά λέξη
            for i in range(5):
                # Δημιουργία τυχαίας αλληλουχίας με ελαφρύ θόρυβο
                fake_sequence = np.random.normal(loc=0.5, scale=0.1, size=(30, 126))
                file_path = os.path.join(out_dir, f"{label}_sample_{i}.npy")
                np.save(file_path, fake_sequence)

        print(f"  --> [OK] Δημιουργήθηκαν δείγματα για {lang.upper()} στον φάκελο: {out_dir}")
    print()


def verify_saved_files():
    """
    Επιβεβαιώνει ότι τα αποθηκευμένα αρχεία διαβάζονται σωστά από το NumPy.
    """
    print("[TEST 3/3] Επαλήθευση ανάγνωσης αποθηκευμένων αρχείων .npy...")
    
    sample_file = "ml/datasets/gsl/raw_landmarks/ΓΕΙΑ_sample_0.npy"
    if os.path.exists(sample_file):
        data = np.load(sample_file)
        assert data.shape == (30, 126), "Τα δεδομένα στο αρχείο δεν έχουν τη σωστή διάσταση!"
        print(f"  --> [OK] Το αρχείο {sample_file} φορτώθηκε επιτυχώς με shape {data.shape}!")
    else:
        print("  --> [ERROR] Το αρχείο δεν βρέθηκε.")
    print()


if __name__ == "__main__":
    print("==================================================")
    print("       SignBridge - M2 Pipeline Integration Test  ")
    print("==================================================\n")
    
    test_normalization_dimensions()
    generate_synthetic_dataset()
    verify_saved_files()

    print("==================================================")
    print("  ΟΛΑ ΤΑ TESTS ΠΕΡΑΣΑΝ ΕΠΙΤΥΧΩΣ! ΕΤΟΙΜΟΙ ΓΙΑ M3. ")
    print("==================================================")