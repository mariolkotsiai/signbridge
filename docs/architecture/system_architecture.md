graph TD
    A[Webcam Stream] -->|MediaStream API| B[Browser React Client]
    B -->|WebSocket / HTTP Frame Stream| C[FastAPI Inference Gateway]
    C -->|Raw Frame / Image| D[MediaPipe Landmark Extractor]
    D -->|Hand + Pose + Face Coordinates| E[Sequence Normalizer & Buffer]
    E -->|Normalized Tensor shape: [T, Keypoints]| F{Profile Selector}
    F -->|GSL Selected| G[GSL PyTorch LSTM/Transformer Model]
    F -->|ASL Selected| H[ASL PyTorch LSTM/Transformer Model]
    G -->|Probability Vector| I[Confidence Filter & Threshold Check]
    H -->|Probability Vector| I
    I -->|> Threshold| J[Predicted Sign Label]
    I -->|< Threshold| K[State: UNKNOWN / UNCERTAIN]
    J --> L[WebSocket JSON Response]
    K --> L
    L -->|Update Subtitles & FPS UI| B