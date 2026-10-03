"""
End-to-end πάνω από πραγματικό TCP: uvicorn + WebSocket (όχι TestClient).
Ο client ακολουθεί ακριβώς το πρωτόκολλο του frontend/src/hooks/useSignStream.ts:
ένα binary JPEG frame "in flight" κάθε φορά, αναμονή απάντησης, JSON control μηνύματα.
"""
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from websockets.sync.client import connect

from services.api.main import create_app
from services.config import PROJECT_ROOT
from services.inference.live_inference import InferenceService, ModelRegistry
from tests.conftest import FakeExtractor


@pytest.fixture()
def server(settings):
    registry = ModelRegistry(settings.weights_dir, settings.languages)
    script = lambda i: "A" if i < 45 else ("B" if i < 100 else None)  # noqa: E731
    service = InferenceService(settings, registry, lambda: FakeExtractor(script))
    app = create_app(service=service)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "uvicorn δεν ξεκίνησε"
    yield f"127.0.0.1:{port}", service
    srv.should_exit = True
    thread.join(timeout=5)


def test_full_session_over_real_socket(server, jpeg_bytes):
    host, service = server

    health = httpx.get(f"http://{host}/api/health").json()
    assert health["models"]["gsl"]["available"] and not health["models"]["asl"]["available"]

    with connect(f"ws://{host}/ws/infer?language=gsl", max_size=2**22) as ws:
        ready = json.loads(ws.recv())
        assert ready["type"] == "ready"

        ws.send(json.dumps({"type": "config", "threshold": 0.6}))
        assert json.loads(ws.recv()) == {"type": "config_ack", "threshold": 0.6, "min_votes": 3}

        committed, states, last = [], [], None
        for _ in range(130):  # ίδιο με το useSignStream: send -> wait for reply -> send
            ws.send(jpeg_bytes)
            msg = json.loads(ws.recv())
            assert msg["type"] == "result"
            states.append(msg["state"])
            p = msg["prediction"]
            if p and p["committed"]:
                committed.append(p["stable_label"])
            last = msg

        # Νόημα A, μετά B χωρίς να κατεβούν τα χέρια, μετά χέρια κάτω => idle.
        assert committed[0] == "A" and committed[-1] == "B"
        assert "sign" in states and states[-1] == "idle"
        assert last["buffer"]["filled"] == 0 and last["hands"] == []

        ws.send(json.dumps({"type": "reset"}))
        assert json.loads(ws.recv())["type"] == "config_ack"

    deadline = time.time() + 3
    while service.active_sessions and time.time() < deadline:
        time.sleep(0.02)
    assert service.active_sessions == 0


def test_close_code_reaches_client_when_model_missing(server):
    host, _ = server
    with connect(f"ws://{host}/ws/infer?language=asl") as ws:
        err = json.loads(ws.recv())
        assert err["code"] == "MODEL_NOT_FOUND"
        with pytest.raises(Exception) as exc:
            ws.recv()
        assert getattr(exc.value, "rcvd", None) is not None and exc.value.rcvd.code == 4404


@pytest.mark.skipif(not (PROJECT_ROOT / "frontend" / "dist" / "index.html").is_file(),
                    reason="Το frontend δεν έχει γίνει build (npm run build)")
def test_built_frontend_is_served_by_fastapi(server):
    host, _ = server
    res = httpx.get(f"http://{host}/")
    assert res.status_code == 200 and "<div id=\"root\">" in res.text
