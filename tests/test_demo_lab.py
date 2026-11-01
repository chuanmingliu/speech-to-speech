"""Demo lab: local URL default, /lab page, and SLO grading helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from server import DEFAULT_SPEECH_TO_SPEECH_URL, resolve_direct_s2s_url  # noqa: E402


def test_resolve_direct_s2s_url_defaults_when_neither_env_is_set():
    assert resolve_direct_s2s_url("", "") == "ws://127.0.0.1:8765/v1/realtime"
    assert DEFAULT_SPEECH_TO_SPEECH_URL == "ws://127.0.0.1:8765/v1/realtime"


def test_resolve_direct_s2s_url_honours_explicit_url():
    assert (
        resolve_direct_s2s_url("ws://example.local:9000/v1/realtime", "")
        == "ws://example.local:9000/v1/realtime"
    )


def test_resolve_direct_s2s_url_defers_to_load_balancer():
    assert resolve_direct_s2s_url("", "http://lb.example") == ""


def test_resolve_direct_s2s_url_explicit_wins_over_lb():
    assert (
        resolve_direct_s2s_url("ws://127.0.0.1:8765/v1/realtime", "http://lb.example")
        == "ws://127.0.0.1:8765/v1/realtime"
    )


@pytest.fixture
def demo_client(monkeypatch):
    monkeypatch.delenv("SPEECH_TO_SPEECH_URL", raising=False)
    monkeypatch.delenv("LOAD_BALANCER_URL", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    import importlib

    import server as demo_server

    importlib.reload(demo_server)
    from fastapi.testclient import TestClient

    return TestClient(demo_server.app)


def test_lab_route_serves_the_demo_page(demo_client):
    res = demo_client.get("/lab")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert 'id="lab-panel"' in res.text
    assert "Monitor" in res.text


def test_config_defaults_to_local_realtime_url(demo_client):
    res = demo_client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["s2sUrl"] == "ws://127.0.0.1:8765/v1/realtime"
    assert body["allowDirect"] is True
    assert body["lb"] is False
    assert body["auth"] is False


def test_metrics_js_locks_slo_ceilings():
    text = (Path(__file__).resolve().parents[1] / "demo" / "ui" / "metrics.js").read_text()
    assert "p50: 700" in text
    assert "p95: 1100" in text
    assert "hard: 1200" in text
    assert "p50: 120" in text
    assert "p95: 250" in text
    assert "p50: 64" in text
    assert "cap: 70" in text
    assert "never invents" in text
    assert "never an SLO certification" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_metrics_js_grades_named_cases():
    script = Path(__file__).resolve().parent / "test_demo_lab_metrics.mjs"
    subprocess.run(["node", str(script)], check=True)
