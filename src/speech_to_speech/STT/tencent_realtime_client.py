from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import numbers
import queue
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlencode

import numpy as np
from websockets.sync.client import connect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TencentRealtimeConfig:
    app_id: str
    secret_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    engine: str = "16k_zh"
    endpoint: str = "asr.cloud.tencent.com"
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 1.0
    write_timeout_s: float = 2.0
    final_timeout_s: float = 10.0
    close_timeout_s: float = 2.0
    max_frame_bytes: int = 6400
    max_json_bytes: int = 1024 * 1024
    max_transcript_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for name in ("app_id", "secret_id", "secret_key", "engine", "endpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Tencent realtime ASR {name} must be a non-empty string.")
        if self.endpoint != "asr.cloud.tencent.com":
            raise ValueError("Tencent realtime ASR endpoint must be the canonical provider hostname.")
        for name in (
            "connect_timeout_s",
            "read_timeout_s",
            "write_timeout_s",
            "final_timeout_s",
            "close_timeout_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Tencent realtime ASR {name} must be a finite positive number.")
        if not 2 <= self.max_frame_bytes <= 6400 or self.max_frame_bytes % 2:
            raise ValueError("Tencent realtime ASR max_frame_bytes must be an even value between 2 and 6400.")
        if not 0 < self.max_json_bytes <= 1024 * 1024:
            raise ValueError("Tencent realtime ASR max_json_bytes must be between 1 and 1048576.")
        if not 0 < self.max_transcript_bytes <= 1024 * 1024:
            raise ValueError("Tencent realtime ASR max_transcript_bytes must be between 1 and 1048576.")


@dataclass(frozen=True)
class TencentRecognitionResult:
    text: str
    final: bool
    stable: bool


def build_tencent_realtime_url(
    config: TencentRealtimeConfig,
    *,
    voice_id: str,
    now_s: int,
    nonce: int,
) -> str:
    if not voice_id:
        raise ValueError("Tencent realtime ASR voice_id must be non-empty.")
    params = {
        "engine_model_type": config.engine,
        "expired": now_s + 3600,
        "filter_empty_result": 1,
        "needvad": 1,
        "nonce": nonce,
        "secretid": config.secret_id,
        "timestamp": now_s,
        "voice_format": 1,
        "voice_id": voice_id,
    }
    canonical_query = urlencode(sorted(params.items()))
    authority_path = f"{config.endpoint}/asr/v2/{config.app_id}"
    signature = base64.b64encode(
        hmac.new(
            config.secret_key.encode(),
            f"{authority_path}?{canonical_query}".encode(),
            hashlib.sha1,
        ).digest()
    ).decode()
    return f"wss://{authority_path}?{canonical_query}&{urlencode({'signature': signature})}"


_END = object()


class TencentRealtimeSession:
    """One bounded Tencent realtime-recognition stream."""

    def __init__(
        self,
        config: TencentRealtimeConfig,
        *,
        voice_id: str | None = None,
        connect_fn: Callable[..., Any] = connect,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.voice_id = voice_id or uuid.uuid4().hex
        self._clock = clock
        self._sleep = sleep
        self._cancelled = threading.Event()
        self._terminal = threading.Event()
        self._writer_done = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()
        self._socket_close_lock = threading.Lock()
        self._socket_closed = False
        self._state_lock = threading.RLock()
        self._frames: queue.Queue[bytes | object] = queue.Queue(maxsize=32)
        self._results: queue.Queue[TencentRecognitionResult] = queue.Queue(maxsize=128)
        self._partials: dict[int, str] = {}
        self._stable: dict[int, str] = {}
        self._transcript_bytes = 0
        self._last_index = -1
        self._last_slice_type = -1
        self._error: RuntimeError | None = None
        self.samples_sent = 0
        url = build_tencent_realtime_url(
            config,
            voice_id=self.voice_id,
            now_s=int(time.time()),
            nonce=secrets.randbelow(2**31),
        )
        try:
            self._websocket = connect_fn(
                url,
                open_timeout=config.connect_timeout_s,
                close_timeout=config.close_timeout_s,
                max_size=config.max_json_bytes,
                compression=None,
            )
        except Exception:
            raise RuntimeError("Tencent realtime ASR connection failed.") from None
        self._writer = threading.Thread(target=self._writer_loop, name="tencent-asr-writer", daemon=True)
        self._reader = threading.Thread(target=self._reader_loop, name="tencent-asr-reader", daemon=True)
        self._writer.start()
        self._reader.start()

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(samples)):
            raise ValueError("Tencent realtime ASR input contains non-finite audio samples.")
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def push_snapshot(self, audio: np.ndarray) -> None:
        self._raise_if_failed()
        samples = np.asarray(audio).reshape(-1)
        if len(samples) < self.samples_sent:
            raise ValueError("Tencent realtime ASR cumulative audio snapshot cannot shrink.")
        unseen = samples[self.samples_sent :]
        pcm = self._to_pcm16(unseen)
        frame_bytes = self.config.max_frame_bytes
        for offset in range(0, len(pcm), frame_bytes):
            self._put_frame(bytes(pcm[offset : offset + frame_bytes]))
        self.samples_sent = len(samples)

    def finish(self, audio: np.ndarray) -> None:
        self.push_snapshot(audio)
        self._put_frame(_END)
        audio_wait_s = len(np.asarray(audio).reshape(-1)) / 16000 + self.config.write_timeout_s
        if not self._writer_done.wait(audio_wait_s):
            self._fail("Tencent realtime ASR writer deadline exceeded.")
        self._raise_if_failed()
        if not self._terminal.wait(self.config.final_timeout_s):
            self._fail("Tencent realtime ASR final result deadline exceeded.")
        self._raise_if_failed()

    def drain_results(self) -> list[TencentRecognitionResult]:
        drained: list[TencentRecognitionResult] = []
        while True:
            try:
                drained.append(self._results.get_nowait())
            except queue.Empty:
                return drained

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.set()
            self._close_socket()
            try:
                self._frames.put_nowait(_END)
            except queue.Full:
                pass
        deadline = monotonic() + self.config.close_timeout_s
        for thread in (self._writer, self._reader):
            remaining = deadline - monotonic()
            if remaining > 0:
                thread.join(remaining)
        self._clear_queue(self._frames)
        self._clear_queue(self._results)
        with self._state_lock:
            self._partials.clear()
            self._stable.clear()
            self._transcript_bytes = 0

    def _put_frame(self, frame: bytes | object) -> None:
        if self._cancelled.is_set():
            self._raise_if_failed()
            raise RuntimeError("Tencent realtime ASR session is closed.")
        try:
            self._frames.put(frame, timeout=self.config.write_timeout_s)
        except queue.Full as exc:
            raise RuntimeError("Tencent realtime ASR outbound queue deadline exceeded.") from exc

    def _writer_loop(self) -> None:
        origin: float | None = None
        samples_written = 0
        try:
            while not self._cancelled.is_set():
                try:
                    frame = self._frames.get(timeout=self.config.read_timeout_s)
                except queue.Empty:
                    continue
                if frame is _END:
                    self._websocket.send('{"type":"end"}')
                    return
                assert isinstance(frame, bytes)
                if origin is None:
                    origin = self._clock()
                target = origin + samples_written / 16000
                delay = target - self._clock()
                if delay > 0:
                    self._sleep(delay)
                self._websocket.send(frame)
                samples_written += len(frame) // 2
        except Exception:
            if not self._cancelled.is_set():
                self._fail("Tencent realtime ASR write failed.")
        finally:
            self._writer_done.set()

    def _reader_loop(self) -> None:
        while not self._cancelled.is_set() and not self._terminal.is_set():
            try:
                message = self._websocket.recv(timeout=self.config.read_timeout_s)
            except TimeoutError:
                continue
            except queue.Empty:
                continue
            except Exception:
                if not self._cancelled.is_set():
                    self._fail("Tencent realtime ASR read failed.")
                return
            try:
                self._accept_message(message)
            except Exception:
                self._fail("Tencent realtime ASR returned an invalid provider event.")
                return

    def _accept_message(self, message: object) -> None:
        if not isinstance(message, str):
            raise ValueError("provider message must be text")
        if len(message.encode()) > self.config.max_json_bytes:
            raise ValueError("provider message is too large")
        payload = json.loads(message)
        if not isinstance(payload, dict) or payload.get("code") != 0:
            raise ValueError("provider rejected request")
        if payload.get("voice_id") != self.voice_id:
            raise ValueError("provider voice ID mismatch")

        result = payload.get("result")
        if result is not None:
            if not isinstance(result, dict):
                raise ValueError("provider result must be an object")
            slice_type = result.get("slice_type")
            index = result.get("index")
            text = result.get("voice_text_str", "")
            if slice_type not in (0, 1, 2) or not isinstance(index, int) or index < 0 or not isinstance(text, str):
                raise ValueError("provider result fields are invalid")
            with self._state_lock:
                if index < self._last_index or (index == self._last_index and slice_type < self._last_slice_type):
                    raise ValueError("provider result order regressed")
                self._last_index = index
                self._last_slice_type = slice_type
                if slice_type == 1:
                    self._transcript_bytes -= len(self._partials.get(index, "").encode())
                    self._transcript_bytes += len(text.encode())
                    if self._transcript_bytes > self.config.max_transcript_bytes:
                        raise ValueError("provider transcript state is too large")
                    self._partials[index] = text
                    self._queue_result(TencentRecognitionResult(text, final=False, stable=False))
                elif slice_type == 2:
                    self._transcript_bytes -= len(self._partials.pop(index, "").encode())
                    self._transcript_bytes -= len(self._stable.get(index, "").encode())
                    self._transcript_bytes += len(text.encode())
                    if self._transcript_bytes > self.config.max_transcript_bytes:
                        raise ValueError("provider transcript state is too large")
                    self._stable[index] = text

        final = payload.get("final", 0)
        if final not in (0, 1):
            raise ValueError("provider final flag is invalid")
        if final == 1:
            with self._state_lock:
                final_text = "".join(self._stable[index] for index in sorted(self._stable))
            self._queue_result(TencentRecognitionResult(final_text, final=True, stable=True))
            self._terminal.set()

    def _queue_result(self, result: TencentRecognitionResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full as exc:
            raise ValueError("provider result buffer is full") from exc

    def _fail(self, message: str) -> None:
        with self._state_lock:
            if self._error is None:
                self._error = RuntimeError(message)
        self._cancelled.set()
        self._terminal.set()
        self._close_socket()

    def _close_socket(self) -> None:
        with self._socket_close_lock:
            if self._socket_closed:
                return
            self._socket_closed = True
            try:
                self._websocket.close()
            except Exception:
                pass

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    @staticmethod
    def _clear_queue(items: queue.Queue[Any]) -> None:
        while True:
            try:
                items.get_nowait()
            except queue.Empty:
                return
