from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import queue
import threading
import time
import traceback
from urllib.parse import parse_qs, quote, urlsplit

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.STT.tencent_realtime_client import (
    TencentRealtimeConfig,
    TencentRealtimeSession,
    TencentRecognitionResult,
    build_tencent_realtime_url,
)


class FakeWebSocket:
    def __init__(self):
        self.binary_messages: list[bytes] = []
        self.text_messages: list[str] = []
        self.incoming: queue.Queue[object] = queue.Queue()
        self.closed = False

    def send(self, message, text=None):
        if isinstance(message, bytes):
            self.binary_messages.append(message)
        else:
            self.text_messages.append(message)

    def recv(self, timeout=None):
        message = self.incoming.get(timeout=timeout)
        if isinstance(message, BaseException):
            raise message
        return message

    def close(self):
        self.closed = True
        self.incoming.put(EOFError("closed"))

    def close_socket(self):
        self.close()

    def provider_event(self, payload):
        self.incoming.put(json.dumps(payload, ensure_ascii=False))


class ProgressingBlockedSendWebSocket(FakeWebSocket):
    def __init__(self, *, block_end: bool):
        super().__init__()
        self.block_end = block_end
        self.progress_count = 0
        self.send_finished = threading.Event()

    def send(self, message, text=None):
        should_block = (message == '{"type":"end"}') if self.block_end else isinstance(message, bytes)
        if not should_block:
            return super().send(message, text=text)
        try:
            started = time.monotonic()
            while not self.closed and time.monotonic() - started < 0.2:
                self.progress_count += 1
                time.sleep(0.005)
            if not self.closed:
                raise RuntimeError("production deadline did not abort progressing Tencent write")
        finally:
            self.send_finished.set()


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition was not reached")


def wait_for_results(session, timeout=1.0):
    observed = []

    def drain_one():
        observed.extend(session.drain_results())
        return bool(observed)

    wait_until(drain_one, timeout)
    return observed


def realtime_config(**overrides):
    values = {
        "app_id": "1250000000",
        "secret_id": "sid",
        "secret_key": "sentinel-secret",
        "read_timeout_s": 0.01,
        "final_timeout_s": 1.0,
    }
    values.update(overrides)
    return TencentRealtimeConfig(**values)


def provider_result(voice_id, text, *, slice_type, index=0, final=0):
    return {
        "code": 0,
        "message": "success",
        "voice_id": voice_id,
        "final": final,
        "result": {
            "slice_type": slice_type,
            "index": index,
            "voice_text_str": text,
        },
    }


class FakeRealtimeSession:
    def __init__(self):
        self.snapshots: list[np.ndarray] = []
        self.finished: list[np.ndarray] = []
        self.results: list[TencentRecognitionResult] = []
        self.closed = False
        self.push_error: Exception | None = None

    def push_snapshot(self, audio):
        if self.push_error:
            raise self.push_error
        self.snapshots.append(np.asarray(audio).copy())

    def finish(self, audio):
        self.finished.append(np.asarray(audio).copy())

    def drain_results(self):
        drained, self.results = self.results, []
        return drained

    def close(self):
        self.closed = True


class FakeSessionFactory:
    def __init__(self):
        self.configs: list[TencentRealtimeConfig] = []
        self.sessions: list[FakeRealtimeSession] = []

    def __call__(self, config):
        self.configs.append(config)
        session = FakeRealtimeSession()
        self.sessions.append(session)
        return session


def make_handler(factory, **setup_overrides):
    from queue import Queue
    from threading import Event

    setup = {
        "app_id": "1250000000",
        "secret_id": "sid",
        "secret_key": "sentinel-secret",
        "session_factory": factory,
    }
    setup.update(setup_overrides)
    return TencentASRHandler(Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs=setup)


def test_tencent_url_signing_uses_the_exact_canonical_query_without_exposing_secret(caplog):
    config = TencentRealtimeConfig(
        app_id="1250000000",
        secret_id="sid",
        secret_key="sentinel-secret",
    )

    url = build_tencent_realtime_url(config, voice_id="voice-1", now_s=1000, nonce=7)

    canonical_query = (
        "engine_model_type=16k_zh&expired=4600&filter_empty_result=1&needvad=1&nonce=7"
        "&secretid=sid&timestamp=1000&voice_format=1&voice_id=voice-1"
    )
    signing_payload = f"asr.cloud.tencent.com/asr/v2/1250000000?{canonical_query}"
    expected_signature = base64.b64encode(
        hmac.new(b"sentinel-secret", signing_payload.encode(), hashlib.sha1).digest()
    ).decode()
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "asr.cloud.tencent.com"
    assert query["voice_format"] == ["1"]
    assert query["signature"] == [expected_signature]
    assert f"signature={quote(expected_signature, safe='')}" in url
    assert "sentinel-secret" not in url + repr(config) + caplog.text


def test_connection_failure_formatted_traceback_cannot_expose_the_signed_url(caplog):
    config = realtime_config(secret_id="sentinel-secret-id")

    def reject_connection(url, **_kwargs):
        raise OSError(f"dial failed for {url}")

    with pytest.raises(RuntimeError) as raised:
        TencentRealtimeSession(config, connect_fn=reject_connection)

    with caplog.at_level(logging.ERROR):
        try:
            raise raised.value
        except RuntimeError:
            logging.getLogger("test.tencent.connection").exception("Tencent connection failed")

    formatted = "".join(traceback.format_exception(raised.value)) + logging.Formatter().format(caplog.records[-1])
    assert raised.value.__cause__ is None
    assert "sentinel-secret-id" not in formatted
    assert "secretid=" not in formatted
    assert "signature=" not in formatted
    assert "wss://" not in formatted


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"app_id": ""}, "app_id"),
        ({"secret_id": ""}, "secret_id"),
        ({"secret_key": ""}, "secret_key"),
        ({"engine": ""}, "engine"),
        ({"endpoint": "https://example.com/path"}, "endpoint"),
        ({"endpoint": "example.com?x=1"}, "endpoint"),
        ({"endpoint": "example.com#fragment"}, "endpoint"),
        ({"endpoint": "user@example.com"}, "endpoint"),
        ({"endpoint": "asr.cloud.tencent.com:443"}, "endpoint"),
        ({"endpoint": "asr.cloud.tencent.com\n"}, "endpoint"),
        ({"endpoint": "ASR.CLOUD.TENCENT.COM"}, "endpoint"),
        ({"connect_timeout_s": 0}, "connect_timeout_s"),
        ({"read_timeout_s": 0}, "read_timeout_s"),
        ({"write_timeout_s": 0}, "write_timeout_s"),
        ({"final_timeout_s": 0}, "final_timeout_s"),
        ({"close_timeout_s": 0}, "close_timeout_s"),
        ({"connect_timeout_s": True}, "connect_timeout_s"),
        ({"read_timeout_s": float("nan")}, "read_timeout_s"),
        ({"write_timeout_s": float("inf")}, "write_timeout_s"),
        ({"final_timeout_s": float("-inf")}, "final_timeout_s"),
        ({"close_timeout_s": "1"}, "close_timeout_s"),
        ({"max_frame_bytes": 1}, "max_frame_bytes"),
        ({"max_frame_bytes": 6401}, "max_frame_bytes"),
        ({"max_json_bytes": 1024 * 1024 + 1}, "max_json_bytes"),
    ],
)
def test_tencent_config_rejects_unbounded_or_missing_connection_values_without_leaking_secret(
    overrides, message
):
    values = {
        "app_id": "1250000000",
        "secret_id": "sid",
        "secret_key": "sentinel-secret",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message) as raised:
        TencentRealtimeConfig(**values)

    assert "sentinel-secret" not in str(raised.value)


def test_session_sends_only_each_snapshot_suffix_in_paced_bounded_frames():
    websocket = FakeWebSocket()
    clock = FakeClock()
    session = TencentRealtimeSession(
        realtime_config(),
        voice_id="voice-stream",
        connect_fn=lambda *_args, **_kwargs: websocket,
        clock=clock,
        sleep=clock.sleep,
    )
    audio = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)

    session.push_snapshot(audio[:3200])
    session.push_snapshot(audio[:6400])
    session.push_snapshot(audio)
    websocket.provider_event(provider_result("voice-stream", "你好", slice_type=2, final=1))
    session.finish(audio)

    expected_pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    assert b"".join(websocket.binary_messages) == expected_pcm
    assert all(len(frame) <= 6400 for frame in websocket.binary_messages)
    assert clock.sleeps == pytest.approx([0.2, 0.2])
    assert websocket.text_messages[-1] == '{"type":"end"}'
    session.close()


@pytest.mark.parametrize("block_end", [False, True], ids=["pcm", "end"])
def test_each_tencent_send_has_total_deadline_and_aborts_progressing_write(block_end):
    websocket = ProgressingBlockedSendWebSocket(block_end=block_end)
    session = TencentRealtimeSession(
        realtime_config(write_timeout_s=0.03),
        voice_id="voice-write-deadline",
        connect_fn=lambda *_args, **_kwargs: websocket,
    )
    audio = np.zeros(3200 if not block_end else 0, dtype=np.float32)
    started = time.monotonic()

    if block_end:
        websocket.provider_event({"code": 0, "voice_id": "voice-write-deadline", "final": 1})
        with pytest.raises(RuntimeError, match="write deadline"):
            session.finish(audio)
    else:
        session.push_snapshot(audio)
        wait_until(lambda: websocket.closed)
        with pytest.raises(RuntimeError, match="write deadline"):
            session.push_snapshot(audio)

    assert time.monotonic() - started < 0.15
    assert websocket.progress_count >= 2
    assert websocket.send_finished.wait(timeout=0.05)
    session.close()


def test_partial_is_observable_before_finish_and_stable_text_becomes_the_only_final_result():
    websocket = FakeWebSocket()
    session = TencentRealtimeSession(
        realtime_config(),
        voice_id="voice-results",
        connect_fn=lambda *_args, **_kwargs: websocket,
    )

    websocket.provider_event(provider_result("voice-results", "你好", slice_type=1))
    assert wait_for_results(session) == [TencentRecognitionResult("你好", final=False, stable=False)]

    websocket.provider_event(provider_result("voice-results", "你好。", slice_type=2))
    websocket.provider_event(
        {
            "code": 0,
            "message": "success",
            "voice_id": "voice-results",
            "final": 1,
        }
    )
    session.finish(np.zeros(0, dtype=np.float32))

    assert session.drain_results() == [TencentRecognitionResult("你好。", final=True, stable=True)]
    session.close()


@pytest.mark.parametrize(
    "event",
    [
        b"binary-is-invalid",
        "{not-json",
        json.dumps({"code": 1001, "message": "sentinel-secret provider detail"}),
        json.dumps({"code": 0, "voice_id": "wrong-voice", "final": 1}),
    ],
)
def test_session_rejects_invalid_provider_events_without_leaking_payload(event):
    websocket = FakeWebSocket()
    session = TencentRealtimeSession(
        realtime_config(),
        voice_id="voice-errors",
        connect_fn=lambda *_args, **_kwargs: websocket,
    )
    websocket.incoming.put(event)

    with pytest.raises(RuntimeError) as raised:
        session.finish(np.zeros(0, dtype=np.float32))

    assert "sentinel-secret" not in str(raised.value)
    session.close()


def test_reader_failure_immediately_closes_socket_without_waiting_for_another_snapshot():
    websocket = FakeWebSocket()
    session = TencentRealtimeSession(
        realtime_config(),
        voice_id="voice-async-failure",
        connect_fn=lambda *_args, **_kwargs: websocket,
    )

    websocket.provider_event({"code": 1001, "message": "provider rejected request"})

    wait_until(lambda: websocket.closed)
    session.close()


def test_provider_transcript_state_closes_session_when_aggregate_utf8_bound_is_exceeded():
    websocket = FakeWebSocket()
    session = TencentRealtimeSession(
        realtime_config(max_transcript_bytes=6),
        voice_id="voice-transcript-bound",
        connect_fn=lambda *_args, **_kwargs: websocket,
    )

    websocket.provider_event(provider_result("voice-transcript-bound", "你好", slice_type=2, index=0))
    websocket.provider_event(provider_result("voice-transcript-bound", "a", slice_type=2, index=1))

    wait_until(lambda: websocket.closed)
    with pytest.raises(RuntimeError, match="invalid provider event"):
        session.finish(np.zeros(0, dtype=np.float32))
    session.close()


def test_handler_reuses_one_session_for_progressive_and_final_snapshots():
    factory = FakeSessionFactory()
    handler = make_handler(factory)
    first = VADAudio(
        audio=np.zeros(3200, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )

    assert list(handler.process(first)) == []
    session = factory.sessions[0]
    session.results = [TencentRecognitionResult("你", final=False, stable=False)]
    progressive = VADAudio(
        audio=np.zeros(6400, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )
    partials = list(handler.process(progressive))

    assert len(factory.sessions) == 1
    assert len(partials) == 1
    assert partials[0].text == "你"
    assert partials[0].turn_id == "turn-1"
    assert partials[0].turn_revision == 0
    assert partials[0].first_partial_at_s is not None
    assert math.isfinite(partials[0].first_partial_at_s)

    session.results = [TencentRecognitionResult("你好。", final=True, stable=True)]
    final = VADAudio(
        audio=np.zeros(8000, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
        created_at_s=123.0,
    )
    transcriptions = list(handler.process(final))

    assert len(session.finished) == 1
    np.testing.assert_array_equal(session.finished[0], final.audio)
    assert len(transcriptions) == 1
    assert transcriptions[0].text == "你好。"
    assert transcriptions[0].language_code == "zh"
    assert transcriptions[0].turn_id == "turn-1"
    assert transcriptions[0].turn_revision == 0
    assert transcriptions[0].speech_stopped_at_s == 123.0
    assert transcriptions[0].final_at_s is not None
    assert math.isfinite(transcriptions[0].final_at_s)
    assert transcriptions[0].final_at_s >= partials[0].first_partial_at_s


def test_handler_closes_old_turn_and_closes_failed_or_ended_sessions():
    factory = FakeSessionFactory()
    handler = make_handler(factory)
    turn_1 = VADAudio(audio=np.zeros(1), mode="progressive", turn_id="turn", turn_revision=0)
    turn_2 = VADAudio(audio=np.zeros(1), mode="progressive", turn_id="turn", turn_revision=1)

    list(handler.process(turn_1))
    first = factory.sessions[0]
    list(handler.process(turn_2))
    second = factory.sessions[1]
    assert first.closed is True

    second.push_error = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        list(handler.process(turn_2))
    assert second.closed is True

    list(handler.process(turn_2))
    third = factory.sessions[2]
    handler.on_session_end()
    assert third.closed is True


def test_handler_requires_app_id_from_environment_without_logging_it(monkeypatch, caplog):
    factory = FakeSessionFactory()
    monkeypatch.setenv("TENCENT_ASR_APP_ID", "sentinel-app-id")
    monkeypatch.setenv("TENCENT_ASR_SECRET_ID", "sid")
    monkeypatch.setenv("TENCENT_ASR_SECRET_KEY", "sentinel-secret")

    handler = make_handler(
        factory,
        app_id=None,
        secret_id=None,
        secret_key=None,
    )
    list(handler.process(VADAudio(audio=np.zeros(1), mode="progressive")))

    assert factory.configs[0].app_id == "sentinel-app-id"
    assert "sentinel-app-id" not in caplog.text
    assert "sentinel-secret" not in caplog.text + repr(factory.configs[0])
