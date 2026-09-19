# ADR 001: Hardware Architecture & Framework Selection

## Status
Accepted

## Context
Project requires real-time GSL/ASL sign recognition running locally on a zero-budget setup (€0).
Development environment specs: AMD Ryzen 5 5600 (6-core CPU), AMD Radeon RX 7600 (8GB VRAM), Windows OS.

## Decision
1. **Pipeline Execution:** Use CPU (Ryzen 5600) for MediaPipe Landmark Extraction and PyTorch CPU inference (latency < 10ms per window for lightweight LSTMs).
2. **Framework Stack:** Python 3.11, PyTorch (CPU/ROCm optional), OpenCV, MediaPipe, FastAPI, React + TypeScript + Vite.
3. **Language Model Separation:** GSL and ASL maintain completely distinct feature datasets, model binaries, and class mappings.

## Consequences
- Zero cloud cost.
- High inference speed (30+ FPS capability).
- Strict separation prevents language collision bugs.