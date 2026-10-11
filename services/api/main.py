"""
SignBridge - FastAPI Inference Gateway.

Εκκίνηση (από το root του project):
    uvicorn services.api.main:app --host 127.0.0.1 --port 8000

Endpoints:
    WS   /ws/infer?language=gsl|asl   ροή frames (binary JPEG) -> αποτελέσματα/υπότιτλοι (JSON)
    GET  /api/health                  κατάσταση service + διαθέσιμα μοντέλα
    POST /api/models/reload           επαναφόρτωση βαρών μετά από νέα εκπαίδευση
    POST /api/predict/sequence        πρόβλεψη από έτοιμα landmarks (debug χωρίς κάμερα)

Πρωτόκολλο WebSocket (ένα frame "in flight" κάθε φορά - ο client περιμένει απάντηση
πριν στείλει το επόμενο, άρα δεν υπάρχει συσσώρευση καθυστέρησης):
    server -> {"type":"ready", ...}                  αμέσως μετά το accept
    client -> <binary JPEG>
    server -> {"type":"result", "state", "hands", "prediction", ...}   ΜΙΑ απάντηση ανά frame
              ή {"type":"frame_error", "code", "message"}              (μη μοιραίο)
    client -> {"type":"config","threshold":0.8} | {"type":"reset"} | {"type":"ping"}
    server -> {"type":"error", ...} + close (4400/4404/4429/4500/4501) για μοιραία σφάλματα
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ml.preprocessing.normalize import normalize_sequence
from services.api.schemas import ClientControl, SequencePrediction, SequenceRequest
from services.config import PROJECT_ROOT, Settings
from services.inference.live_inference import (
    BadFrame,
    InferenceService,
    LandmarkerUnavailable,
    ModelLoadError,
    ModelNotAvailable,
    TooManySessions,
    format_top_k,
)

log = logging.getLogger("signbridge.api")

# Κωδικοί κλεισίματος WebSocket (περιοχή 4000-4999 = εφαρμογή).
WS_BAD_REQUEST = 4400
WS_NO_MODEL = 4404
WS_BUSY = 4429
WS_MODEL_ERROR = 4500
WS_LANDMARKER_ERROR = 4501


def create_app(service: Optional[InferenceService] = None, settings: Optional[Settings] = None) -> FastAPI:
    """Factory: επιτρέπει στα tests να περάσουν δικό τους service (π.χ. με ψεύτικο extractor)."""
    settings = settings or (service.settings if service else Settings.from_env())
    svc = service or InferenceService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        app.state.service = svc
        # Προφόρτωση: το πρώτο WebSocket δεν πληρώνει το κόστος φόρτωσης PyTorch.
        for lang, info in svc.registry.describe().items():
            log.info("Μοντέλο %s: %s", lang, "OK" if info["available"] else info.get("error"))
        if not svc.landmarker_available():
            log.warning("Λείπει το %s - τρέξε: python scripts/download_models.py",
                        svc.settings.landmarker_model_path)
        yield
        svc.shutdown()

    app = FastAPI(title="SignBridge Inference Gateway", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ REST
    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "active_sessions": svc.active_sessions,
            "max_sessions": svc.settings.max_sessions,
            "landmarker_model": svc.landmarker_available(),
            "defaults": {"threshold": svc.settings.threshold, "window_frames": svc.settings.window_frames},
            "models": svc.registry.describe(),
        }

    @app.post("/api/models/reload")
    def reload_models() -> dict:
        svc.registry.reload()
        return {"models": svc.registry.describe()}

    @app.post("/api/predict/sequence", response_model=SequencePrediction)
    def predict_sequence(req: SequenceRequest) -> SequencePrediction:
        """Ίδιο μονοπάτι με το live (normalize_sequence -> LSTM), αλλά με landmarks από τον client."""
        try:
            model = svc.registry.get(req.language)
        except ModelNotAvailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        window = normalize_sequence(req.sequence, target_frames=model.seq_len)
        top_k = format_top_k(model.predict_proba(window), model.labels, k=3)
        return SequencePrediction(language=req.language, label=top_k[0]["label"],
                                  confidence=top_k[0]["confidence"], top_k=top_k)

    # ------------------------------------------------------------- WebSocket
    async def _fatal(ws: WebSocket, code: int, error_code: str, message: str) -> None:
        await ws.send_json({"type": "error", "code": error_code, "message": message})
        await ws.close(code=code)

    @app.websocket("/ws/infer")
    async def ws_infer(ws: WebSocket, language: str = Query("gsl")) -> None:
        await ws.accept()
        language = language.lower()
        if language not in svc.settings.languages:
            return await _fatal(ws, WS_BAD_REQUEST, "UNSUPPORTED_LANGUAGE",
                                f"Μη υποστηριζόμενη γλώσσα '{language}'. Επιλογές: {', '.join(svc.settings.languages)}")

        try:
            session = await svc.open_session(language)
        except ModelNotAvailable as exc:
            return await _fatal(ws, WS_NO_MODEL, "MODEL_NOT_FOUND", str(exc))
        except ModelLoadError as exc:
            return await _fatal(ws, WS_MODEL_ERROR, "MODEL_INVALID", str(exc))
        except LandmarkerUnavailable as exc:
            return await _fatal(ws, WS_LANDMARKER_ERROR, "LANDMARKER_MISSING", str(exc))
        except TooManySessions as exc:
            return await _fatal(ws, WS_BUSY, "SERVER_BUSY", str(exc))

        loop = asyncio.get_running_loop()
        try:
            await ws.send_json(session.ready_message())
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                data = msg.get("bytes")
                if data is not None:
                    if len(data) > svc.settings.max_frame_bytes:
                        await ws.send_json({"type": "frame_error", "code": "FRAME_TOO_LARGE",
                                            "message": f"Frame > {svc.settings.max_frame_bytes} bytes"})
                        continue
                    try:
                        # Decode + MediaPipe + LSTM σε worker thread: ο event loop μένει ελεύθερος.
                        result = await loop.run_in_executor(svc.executor, session.process_jpeg, data)
                    except BadFrame as exc:
                        await ws.send_json({"type": "frame_error", "code": "BAD_FRAME", "message": str(exc)})
                        continue
                    except Exception:  # σφάλμα MediaPipe/PyTorch: δεν ρίχνουμε τη σύνδεση
                        log.exception("Σφάλμα επεξεργασίας frame")
                        await ws.send_json({"type": "frame_error", "code": "INFERENCE_FAILED",
                                            "message": "Εσωτερικό σφάλμα επεξεργασίας frame"})
                        continue
                    await ws.send_json(result)
                    continue

                text = msg.get("text")
                if text is not None:
                    try:
                        ctl = ClientControl.model_validate_json(text)
                    except ValidationError as exc:
                        await ws.send_json({"type": "error", "code": "BAD_CONTROL",
                                            "message": exc.errors()[0]["msg"]})
                        continue
                    if ctl.type == "ping":
                        await ws.send_json({"type": "pong"})
                    elif ctl.type == "reset":
                        session.reset()
                        await ws.send_json({"type": "config_ack", **session.runtime_config()})
                    else:  # config
                        session.configure(threshold=ctl.threshold, min_votes=ctl.min_votes)
                        await ws.send_json({"type": "config_ack", **session.runtime_config()})
        except WebSocketDisconnect:
            pass
        finally:
            await svc.close_session(session)

    # ------------------------------------------- Static frontend (προαιρετικό)
    dist = Path(PROJECT_ROOT) / "frontend" / "dist"
    if (dist / "index.html").is_file():
        # Μετά το `npm run build` το UI σερβίρεται από το ίδιο process στο http://127.0.0.1:8000/
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


app = create_app()
