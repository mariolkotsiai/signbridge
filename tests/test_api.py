import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from services.api.main import create_app
from services.config import Settings
from services.inference.live_inference import InferenceService, ModelRegistry
from tests.conftest import LEVELS, FakeExtractor


def build_client(settings, script=lambda i: "A", **overrides):
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})
    registry = ModelRegistry(settings.weights_dir, settings.languages)
    extractors = []

    def factory():
        e = FakeExtractor(script)
        extractors.append(e)
        return e

    service = InferenceService(settings, registry, factory)
    return TestClient(create_app(service=service)), service, extractors


def wait_for(cond, timeout=3.0):
    """Το cleanup του server είναι ασύγχρονο: περιμένουμε (με όριο) μέχρι να ισχύσει η συνθήκη."""
    import time
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def stream(ws, jpeg, n):
    out = []
    for _ in range(n):
        ws.send_bytes(jpeg)
        out.append(ws.receive_json())
    return out


def test_health_reports_models(settings):
    client, _, _ = build_client(settings)
    with client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["models"]["gsl"]["available"] is True
    assert body["models"]["gsl"]["labels"] == ["A", "B"]
    assert body["models"]["asl"]["available"] is False


def test_websocket_streams_results_and_commits_subtitle(settings, jpeg_bytes):
    client, service, extractors = build_client(settings)
    with client:
        with client.websocket_connect("/ws/infer?language=gsl") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["labels"] == ["A", "B"]
            assert ready["window_frames"] == 30
            msgs = stream(ws, jpeg_bytes, 60)
            assert service.active_sessions == 1
        # Μετά το disconnect: ελευθερώνεται το slot και κλείνει ο extractor.
        assert wait_for(lambda: service.active_sessions == 0)
        assert wait_for(lambda: extractors[0].closed)
    assert all(m["type"] == "result" for m in msgs)
    commits = [m["prediction"] for m in msgs if m["prediction"] and m["prediction"]["committed"]]
    assert [c["stable_label"] for c in commits] == ["A"]
    assert commits[0]["confidence"] > 0.7
    assert msgs[-1]["state"] == "sign" and len(msgs[-1]["hands"]) == 1


def test_missing_model_closes_with_clear_error(settings):
    client, service, _ = build_client(settings)
    with client, client.websocket_connect("/ws/infer?language=asl") as ws:
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "MODEL_NOT_FOUND"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4404
    assert wait_for(lambda: service.active_sessions == 0)


def test_unsupported_language(settings):
    client, _, _ = build_client(settings)
    with client, client.websocket_connect("/ws/infer?language=xx") as ws:
        assert ws.receive_json()["code"] == "UNSUPPORTED_LANGUAGE"


def test_bad_frame_is_non_fatal(settings, jpeg_bytes):
    client, _, _ = build_client(settings)
    with client, client.websocket_connect("/ws/infer?language=gsl") as ws:
        ws.receive_json()
        ws.send_bytes(b"garbage")
        assert ws.receive_json() == {"type": "frame_error", "code": "BAD_FRAME",
                                     "message": "Το frame δεν αποκωδικοποιείται ως εικόνα."}
        assert stream(ws, jpeg_bytes, 1)[0]["type"] == "result"  # η σύνδεση συνεχίζει


def test_oversized_frame_rejected(settings, jpeg_bytes):
    client, _, _ = build_client(settings, max_frame_bytes=100)
    with client, client.websocket_connect("/ws/infer?language=gsl") as ws:
        ws.receive_json()
        ws.send_bytes(jpeg_bytes)
        assert ws.receive_json()["code"] == "FRAME_TOO_LARGE"


def test_control_messages(settings, jpeg_bytes):
    client, _, _ = build_client(settings)
    with client, client.websocket_connect("/ws/infer?language=gsl") as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
        ws.send_json({"type": "config", "threshold": 0.9, "min_votes": 2})
        assert ws.receive_json() == {"type": "config_ack", "threshold": 0.9, "min_votes": 2}
        ws.send_json({"type": "config", "threshold": 5})
        assert ws.receive_json()["code"] == "BAD_CONTROL"
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "BAD_CONTROL"
        stream(ws, jpeg_bytes, 20)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "config_ack"
        assert stream(ws, jpeg_bytes, 1)[0]["buffer"]["filled"] == 1


def test_session_limit(settings):
    client, service, _ = build_client(settings, max_sessions=1)
    with client:
        with client.websocket_connect("/ws/infer?language=gsl") as first:
            assert first.receive_json()["type"] == "ready"
            with client.websocket_connect("/ws/infer?language=gsl") as second:
                err = second.receive_json()
                assert err["code"] == "SERVER_BUSY"
            assert service.active_sessions == 1  # η απορριφθείσα σύνδεση δεν κράτησε slot
        assert wait_for(lambda: service.active_sessions == 0)
        with client.websocket_connect("/ws/infer?language=gsl") as again:  # το slot ελευθερώθηκε
            assert again.receive_json()["type"] == "ready"


def test_predict_sequence_endpoint(settings):
    client, _, _ = build_client(settings)
    # Ίδια δομή με τα δεδομένα εκπαίδευσης: 1ο χέρι «A», 2ο χέρι μηδενικά.
    level = LEVELS["A"]
    row = [v for k in range(21) for v in (level, level + 0.02 * k / 21, level * 0.1)] + [0.0] * 63
    with client:
        r = client.post("/api/predict/sequence", json={"language": "gsl", "sequence": [row] * 12})
        assert r.status_code == 200 and r.json()["label"] == "A"
        assert client.post("/api/predict/sequence",
                           json={"language": "gsl", "sequence": [[0.1] * 10]}).status_code == 422
        assert client.post("/api/predict/sequence",
                           json={"language": "asl", "sequence": [row]}).status_code == 404


def test_reload_endpoint(settings):
    client, _, _ = build_client(settings)
    with client:
        body = client.post("/api/models/reload").json()
    assert body["models"]["gsl"]["available"] is True
