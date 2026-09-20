import os
import sys
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Direct MP4 Links (για ASL)
DIRECT_SAMPLES = {
    "asl": {
        "HELLO": [
            "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/video_samples/00100.mp4",
            "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/video_samples/00101.mp4"
        ],
        "THANK_YOU": [
            "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/video_samples/00200.mp4",
            "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/video_samples/00201.mp4"
        ]
    }
}

# 2. YouTube Links (για GSL)
YOUTUBE_SAMPLES = {
    "gsl": {
        "ΓΕΙΑ": [
            "https://www.youtube.com/watch?v=jwf5f2AnAE8&list=PLv-TYzCHI-uasqs-Bl0_rlPqrW7TkQGk3&index=3",
        ],
        "ΕΥΧΑΡΙΣΤΩ": [
            "https://www.youtube.com/watch?v=Yw4Mr0Se_P4&list=PLv-TYzCHI-uasqs-Bl0_rlPqrW7TkQGk3&index=27",
        ]
    }
}


def download_direct_files():
    print("--- 1. Λήψη Direct MP4 Βίντεο (ASL) ---")
    for lang, labels in DIRECT_SAMPLES.items():
        for label, urls in labels.items():
            target_dir = os.path.join(PROJECT_ROOT, f"ml/raw_videos/{lang}/{label}")
            os.makedirs(target_dir, exist_ok=True)
            
            for idx, url in enumerate(urls):
                file_path = os.path.join(target_dir, f"{label}_sample_{idx+1}.mp4")
                if not os.path.exists(file_path):
                    print(f"Κατέβασμα: [{lang.upper()}] {label} ({idx+1}/{len(urls)})...")
                    try:
                        urllib.request.urlretrieve(url, file_path)
                        print(f"  [OK] Αποθηκεύτηκε στο: {file_path}")
                    except Exception as e:
                        print(f"  [ERROR] Αποτυχία λήψης: {e}")
                else:
                    print(f"  [SKIP] Το αρχείο υπάρχει ήδη: {file_path}")


def download_youtube_files():
    print("\n--- 2. Λήψη YouTube Βίντεο (GSL) ---")
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("[WARNING] Το yt-dlp δεν είναι εγκατεστημένο. Τρέξε: pip install yt-dlp")
        return

    for lang, labels in YOUTUBE_SAMPLES.items():
        for label, urls in labels.items():
            if not urls:
                continue
            
            target_dir = os.path.join(PROJECT_ROOT, f"ml/raw_videos/{lang}/{label}")
            os.makedirs(target_dir, exist_ok=True)

            ydl_opts = {
                'format': 'mp4/best',
                'outtmpl': os.path.join(target_dir, '%(id)s.%(ext)s'),
                'quiet': True
            }

            with YoutubeDL(ydl_opts) as ydl:
                for url in urls:
                    try:
                        print(f"Κατέβασμα YouTube [{lang.upper()}]: {label}...")
                        ydl.download([url])
                        print(f"  [OK] Ολοκληρώθηκε.")
                    except Exception as e:
                        print(f"  [ERROR] Αποτυχία λήψης για {url}: {e}")


if __name__ == "__main__":
    download_direct_files()
    download_youtube_files()
    print("\n[SUCCESS] Η διαδικασία λήψης ολοκληρώθηκε!")