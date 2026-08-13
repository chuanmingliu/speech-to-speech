from __future__ import annotations

import io
import json
import ssl
from collections import deque
from queue import Queue
from threading import Event
from typing import Any

import av
import numpy as np
from websockets.frames import CloseCode


def _encoded_mp3(sample_count: int = 16_000) -> tuple[bytes, np.ndarray]:
    time = np.arange(sample_count, dtype=np.float64) / 16_000
    source = np.asarray(np.sin(2 * np.pi * 440 * time) * 12_000, dtype=np.int16)
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp3")
    stream = container.add_stream("mp3", rate=16_000)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(source.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = 16_000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue(), source


def test_decoder_emits_ordered_fixed_pcm_blocks_before_finish() -> None:
    from speech_to_speech.TTS.incremental_mp3_decoder import IncrementalMP3Decoder

    encoded, source = _encoded_mp3()
    decoder = IncrementalMP3Decoder(sample_rate=16_000, channels=1, block_samples=512)
    split = len(encoded) // 2 + 7

    first = decoder.feed(encoded[:split])
    second = decoder.feed(encoded[split:])
    tail = decoder.finish()
    blocks = first + second + tail

    assert first or second
    assert blocks
    assert all(block.dtype == np.int16 and block.shape == (512,) for block in blocks)
    decoded = np.concatenate(blocks)
    assert np.count_nonzero(decoded) > 0
    assert np.max(np.abs(np.diff(decoded.astype(np.int32)))) < 20_000
    assert len(decoded) - np.flatnonzero(decoded)[-1] - 1 < 512
    assert len(decoded) >= len(source)


def test_decoder_rejects_oversized_input_and_closes_idempotently() -> None:
    from speech_to_speech.TTS.incremental_mp3_decoder import IncrementalMP3Decoder

    decoder = IncrementalMP3Decoder()

    with np.testing.assert_raises_regex(ValueError, "1 MiB"):
        decoder.feed(b"x" * (1024 * 1024 + 1))

    decoder.close()
    decoder.close()
    with np.testing.assert_raises_regex(RuntimeError, "closed"):
        decoder.feed(b"")


def _event(event: str | None = None, **fields: Any) -> str:
    body: dict[str, Any] = {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        **fields,
    }
    if event is not None:
        body["event"] = event
    return json.dumps(body)


class FakeClientConnection:
    def __init__(self, received: list[str | Exception]) -> None:
        self.received = deque(received)
        self.sent: list[dict[str, Any]] = []
        self.recv_calls = 0
        self.closed = False

    def recv(self, timeout: float | None = None, decode: bool | None = None) -> str:
        self.recv_calls += 1
        if not self.received:
            raise TimeoutError
        item = self.received.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def send(self, message: Any, text: bool | None = None) -> None:
        self.sent.append(json.loads(message))

    def close(self, code: CloseCode | int = CloseCode.NORMAL_CLOSURE, reason: str = "") -> None:
        self.closed = True


class FakeDecoder:
    def __init__(self, sample_rate: int = 16_000, channels: int = 1, block_samples: int = 512) -> None:
        self.fed: list[bytes] = []
        self.closed = False

    def feed(self, encoded: bytes) -> list[np.ndarray]:
        self.fed.append(encoded)
        value = encoded[0]
        return [np.full(512, value, dtype=np.int16)]

    def finish(self) -> list[np.ndarray]:
        return []

    def close(self) -> None:
        self.closed = True


def _streaming_client(received: list[str | Exception], **config_overrides: Any):
    from speech_to_speech.TTS.minimax_tts_handler import MiniMaxStreamingClient, MiniMaxTTSConfig

    websocket = FakeClientConnection(received)
    connection: dict[str, Any] = {}

    def connect(uri: str, **kwargs: Any) -> FakeClientConnection:
        connection["uri"] = uri
        connection["kwargs"] = kwargs
        return websocket

    config = MiniMaxTTSConfig(api_key="test-key", voice_id="test-voice", **config_overrides)
    client = MiniMaxStreamingClient(config, connect_fn=connect, decoder_factory=FakeDecoder)
    return client, websocket, connection


def test_handshake_uses_authenticated_hardened_websocket_and_official_settings() -> None:
    client, websocket, connection = _streaming_client(
        [_event("connected_success"), _event("task_started")]
    )

    client.start()

    assert connection["uri"] == "wss://api.minimax.io/ws/v1/t2a_v2"
    kwargs = connection["kwargs"]
    assert kwargs["additional_headers"] == {"Authorization": "Bearer test-key"}
    assert isinstance(kwargs["ssl"], ssl.SSLContext)
    assert kwargs["ssl"].check_hostname is True
    assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert kwargs["proxy"] is None
    assert kwargs["compression"] is None
    assert kwargs["max_size"] == 1024 * 1024
    assert kwargs["max_queue"] == 4
    assert kwargs["open_timeout"] > 0
    assert kwargs["close_timeout"] > 0
    start = websocket.sent[0]
    assert start == {
        "event": "task_start",
        "model": "speech-2.8-turbo",
        "language_boost": "auto",
        "voice_setting": {
            "voice_id": "test-voice",
            "speed": 1.1,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 16_000,
            "bitrate": 128_000,
            "format": "mp3",
            "channel": 1,
        },
    }


def test_order_preserves_multiple_audio_fragments_and_requires_clean_finish() -> None:
    client, websocket, _ = _streaming_client(
        [
            _event("connected_success"),
            _event("task_started"),
            _event(data={"audio": "01"}, is_final=False),
            _event(data={"audio": "02"}, is_final=True),
            _event("task_finished"),
        ]
    )
    client.start()

    blocks = list(client.synthesize("hello", cancelled=lambda: False))
    tail = client.finish()

    assert [message["event"] for message in websocket.sent] == [
        "task_start",
        "task_continue",
        "task_finish",
    ]
    assert websocket.sent[1] == {"event": "task_continue", "text": "hello"}
    assert [int(block[0]) for block in blocks] == [1, 2]
    assert tail == []


def test_before_terminal_audio_is_yielded_without_reading_next_event() -> None:
    terminal_released = Event()
    client, websocket, _ = _streaming_client(
        [
            _event("connected_success"),
            _event("task_started"),
            _event(data={"audio": "03"}, is_final=False),
            _event(data={"audio": "04"}, is_final=True),
        ]
    )
    client.start()
    audio_iter = client.synthesize("hello", cancelled=terminal_released.is_set)

    first = next(audio_iter)

    assert first.tolist() == [3] * 512
    assert terminal_released.is_set() is False
    assert websocket.recv_calls == 3


def test_cancel_between_fragments_closes_socket_and_drops_late_audio() -> None:
    cancelled = Event()
    client, websocket, _ = _streaming_client(
        [
            _event("connected_success"),
            _event("task_started"),
            _event(data={"audio": "05"}, is_final=False),
            _event(data={"audio": "06"}, is_final=True),
        ]
    )
    client.start()
    audio_iter = client.synthesize("hello", cancelled=cancelled.is_set)

    assert next(audio_iter).tolist() == [5] * 512
    cancelled.set()

    assert list(audio_iter) == []
    assert websocket.closed is True
    assert websocket.recv_calls == 3


def test_failed_provider_events_and_malformed_payloads_fail_closed() -> None:
    cases = [
        _event("task_failed", base_resp={"status_code": 1004, "status_msg": "rejected"}),
        _event(data={"audio": "not-hex"}, is_final=True),
        _event("task_finished"),
        _event(data={"audio": "01"}),
    ]
    for provider_event in cases:
        client, websocket, _ = _streaming_client(
            [_event("connected_success"), _event("task_started"), provider_event]
        )
        client.start()

        with np.testing.assert_raises(RuntimeError):
            list(client.synthesize("hello", cancelled=lambda: False))

        assert websocket.closed is True


def test_oversized_provider_event_and_audio_fail_closed() -> None:
    oversized_event = "{" + (" " * (1024 * 1024)) + "}"
    oversized_audio = _event(data={"audio": "00" * 17}, is_final=True)
    for provider_event, overrides in (
        (oversized_event, {}),
        (oversized_audio, {"max_audio_bytes": 16}),
    ):
        client, websocket, _ = _streaming_client(
            [_event("connected_success"), _event("task_started"), provider_event],
            **overrides,
        )
        client.start()

        with np.testing.assert_raises(RuntimeError):
            list(client.synthesize("hello", cancelled=lambda: False))

        assert websocket.closed is True


def test_failed_read_timeout_closes_socket() -> None:
    client, websocket, _ = _streaming_client(
        [_event("connected_success"), _event("task_started"), TimeoutError()],
        read_poll_timeout_s=0.0001,
        event_timeout_s=0.001,
    )
    client.start()

    with np.testing.assert_raises(TimeoutError):
        list(client.synthesize("hello", cancelled=lambda: False))

    assert websocket.closed is True


def test_reuse_keeps_one_task_per_response_and_starts_fresh_after_done() -> None:
    from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
    from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler

    sockets = deque(
        [
            FakeClientConnection(
                [
                    _event("connected_success"),
                    _event("task_started"),
                    _event(data={"audio": "07"}, is_final=True),
                    _event(data={"audio": "08"}, is_final=True),
                    _event("task_finished"),
                ]
            ),
            FakeClientConnection(
                [
                    _event("connected_success"),
                    _event("task_started"),
                    _event(data={"audio": "09"}, is_final=True),
                    _event("task_finished"),
                ]
            ),
        ]
    )
    created: list[Any] = []

    def client_factory(config: Any):
        from speech_to_speech.TTS.minimax_tts_handler import MiniMaxStreamingClient

        websocket = sockets.popleft()
        client = MiniMaxStreamingClient(config, connect_fn=lambda uri, **kwargs: websocket, decoder_factory=FakeDecoder)
        created.append((client, websocket))
        return client

    handler = MiniMaxTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "api_key": "test-key",
            "voice_id": "test-voice",
            "client_factory": client_factory,
        },
    )

    first = list(handler.process(TTSInput(text="first")))
    second = list(handler.process(TTSInput(text="second")))
    done = list(handler.process(EndOfResponse()))
    third = list(handler.process(TTSInput(text="new response")))
    handler.on_session_end()
    handler.cleanup()

    assert [int(block[0]) for block in first + second + third] == [7, 8, 9]
    assert done == [AUDIO_RESPONSE_DONE]
    assert len(created) == 2
    assert [[message["event"] for message in websocket.sent] for _, websocket in created] == [
        ["task_start", "task_continue", "task_continue", "task_finish"],
        ["task_start", "task_continue"],
    ]
    assert all(websocket.closed for _, websocket in created)
