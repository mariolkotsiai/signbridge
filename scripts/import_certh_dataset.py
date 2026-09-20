import os
import shutil
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- ΠΡΟΣΑΡΜΟΣΕ ΑΥΤΑ ΤΑ PATHS ---
EXTRACTED_VIDEOS_DIR = r"C:\Users\mario\Desktop\theProject\health1"  # Ο φάκελος που περιέχει τους υποφακέλους με τα PNG
CSV_ANNOTATION_FILE = r"C:\Users\mario\Desktop\theProject\GSL_split\GSL_isolated\train_greek_iso.csv"
# ----------------------------------

TARGET_RAW_DIR = os.path.join(PROJECT_ROOT, "ml", "raw_videos", "gsl")


def auto_organize_png_certh():
    if not os.path.exists(EXTRACTED_VIDEOS_DIR):
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος: {EXTRACTED_VIDEOS_DIR}")
        return

    if not os.path.exists(CSV_ANNOTATION_FILE):
        print(f"[ERROR] Δεν βρέθηκε το CSV: {CSV_ANNOTATION_FILE}")
        return

    print("--- Αυτόματη Οργάνωση CERTH (PNG Sequences) ---")
    
    # Διαβάζουμε το CSV χωρίς headers και με διαχωριστικό '|'
    df = pd.read_csv(CSV_ANNOTATION_FILE, sep='|', header=None, names=['path', 'label'])

    copied_folders = 0
    missing_folders = 0

    for _, row in df.iterrows():
        raw_path = str(row['path']).strip()
        label = str(row['label']).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Εξαγωγή μόνο του ονόματος του φακέλου (π.χ. glosses0000)
        folder_name = os.path.basename(raw_path)
        folder_name = os.path.splitext(folder_name)[0]

        src_folder = os.path.join(EXTRACTED_VIDEOS_DIR, folder_name)

        if os.path.exists(src_folder) and os.path.isdir(src_folder):
            target_label_dir = os.path.join(TARGET_RAW_DIR, label, folder_name)
            
            if not os.path.exists(target_label_dir):
                shutil.copytree(src_folder, target_label_dir)
                copied_folders += 1
        else:
            missing_folders += 1

    print(f"\n[SUCCESS] Ολοκληρώθηκε!")
    print(f" -> Αντιγράφηκαν: {copied_folders} δείγματα/φακέλοι στο ml/raw_videos/gsl/")
    if missing_folders > 0:
        print(f" -> Δεν βρέθηκαν στο health1: {missing_folders} δείγματα (είναι φυσιολογικό αν το health1 περιέχει υποσύνολο του dataset).")


if __name__ == "__main__":
    auto_organize_png_certh()