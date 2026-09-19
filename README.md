# SignBridge 

Local-first proof-of-concept for real-time sign language recognition (Greek Sign Language - GSL & American Sign Language - ASL) using a webcam stream and lightweight deep learning models.

## Project Scope (M0–M6 Prototype)
- **Target Languages:** GSL (ΕΝΓ) and ASL (Explicitly separated profiles).
- **Architecture:** Webcam → MediaPipe Landmark Extraction → Temporal Classifier → FastAPI Backend → React Web Subtitles UI.
- **Budget Constraint:** €0 (100% Free / Open Source / Local Execution).

## Hardware & Stack
- **CPU:** AMD Ryzen 5 5600 (6-Core)
- **GPU:** AMD Radeon RX 7600 (8GB VRAM)
- **Backend:** Python 3.11, PyTorch, OpenCV, MediaPipe, FastAPI
- **Frontend:** React, TypeScript, Vite