from __future__ import annotations

import io
import logging
import os
import re
import unicodedata
import wave
from collections import OrderedDict
from collections.abc import Callable
from threading import Event, Lock, RLock
from time import perf_counter
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
from rich.console import Console
from websockets.exceptions import ConnectionClosed

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.minimax_websocket import (
    DEFAULT_OPEN_TIMEOUT_S,
    MiniMaxProviderError,
    MiniMaxWebSocketSession,
)

logger = logging.getLogger(__name__)
console = Console()

_DEFAULT_CACHE_MB = 32.0
_DEFAULT_KEEPALIVE_S = 300.0
_WARMUP_TIMEOUT_S = 5.0
_CONNECTION_PROBE_INTERVAL_S = 30.0
_WEBSOCKET_MAX_IDLE_S = 90.0
# A failed turn is already producing no audio; do not also block the handler
# thread for long probing whether the session survived.
_FAILED_SESSION_PROBE_S = 0.25
_DEFAULT_MODEL_WARMUP_TEXT = "Hi."


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _has_speakable_content(text: str) -> bool:
    """Return True when text contains at least one letter or digit to pronounce.

    MiniMax accepts punctuation-only ``task_continue`` payloads and returns
    ``is_final`` with empty audio; treat those as silent instead of errors.
    """
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            return True
    return False


def _parse_prime_texts(value: str | list[str] | None) -> tuple[str, ...]:
    """Normalise the prime list from a config list or a delimited env string."""
    if not value:
        return ()
    items = value if isinstance(value, list) else re.split(r"[\n|]", value)
    seen: dict[str, None] = {}
    for item in items:
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


def _http_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme not in {"ws", "wss"}:
        return endpoint
    # There is no bidi one-shot HTTP route; both WebSocket paths map back to the
    # single /v1/t2a_v2 endpoint used by the non-streaming fallback.
    path = parts.path.replace("/ws/v1/t2a_v2_bidi", "/v1/t2a_v2").replace("/ws/v1/t2a_v2", "/v1/t2a_v2")
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


def _websocket_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme in {"ws", "wss"}:
        # An explicit ws(s) endpoint is honoured as given, so pinning the older
        # /ws/v1/t2a_v2 path still works.
        return endpoint
    if parts.scheme not in {"http", "https"}:
        raise ValueError("MiniMax TTS endpoint must use http(s) or ws(s).")
    path = parts.path.replace("/v1/t2a_v2", "/ws/v1/t2a_v2_bidi")
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


class _AudioLRUCache:
    """Thread-safe process cache shared by telephony pipeline lanes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._bytes = 0
        self._items: OrderedDict[tuple[Any, ...], tuple[bytes, ...]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> tuple[bytes, ...] | None:
        with self._lock:
            cached = self._items.pop(key, None)
            if cached is not None:
                self._items[key] = cached
            return cached

    def put(self, key: tuple[Any, ...], chunks: list[bytes], max_bytes: int) -> None:
        if max_bytes <= 0 or not chunks:
            return
        size = sum(len(chunk) for chunk in chunks)
        if size > max_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= sum(len(chunk) for chunk in previous)
            stored = tuple(chunks)
            self._items[key] = stored
            self._bytes += size
            while self._bytes > max_bytes and self._items:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= sum(len(chunk) for chunk in evicted)


_SHARED_AUDIO_CACHE = _AudioLRUCache()


class MiniMaxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """MiniMax T2A adapter producing 16 kHz mono PCM16 chunks.

    Streaming uses one persistent WebSocket task per pipeline lane, following
    MiniMax's ``task_start`` / ``task_continue`` protocol. Set
    ``MINIMAX_TTS_STREAM=false`` to fall back to one HTTP WAV response.
    """

    def setup(
        self,
        should_listen: Event,
        api_key: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        speed: float | None = None,
        endpoint: str | None = None,
        websocket_endpoint: str | None = None,
        language_boost: str | None = None,
        sample_rate: int = 16000,
        blocksize: int = 512,
        request_timeout_s: float = 30.0,
        stream: bool | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        client: Any | None = None,
        websocket_connect: Callable[..., Any] | None = None,
        websocket_open_timeout_s: float | None = None,
        websocket_receive_timeout_s: float | None = None,
        websocket_max_idle_s: float | None = None,
        connection_keepalive_s: float | None = None,
        warmup_connection: bool | None = None,
        warmup_model: bool | None = None,
        model_warmup_text: str | None = None,
        prime_texts: str | list[str] | None = None,
        cache_max_mb: float | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.api_key = api_key or os.getenv("MINIMAX_TTS_API_KEY")
        self.model = model or os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-turbo")
        self.voice_id = voice_id or os.getenv("MINIMAX_TTS_VOICE_ID")
        self.speed = (
            _env_float("MINIMAX_TTS_SPEED", 1.0)
            if speed is None
            else float(speed)
        )
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("MiniMax TTS speed must be between 0.5 and 2.0.")
        configured_endpoint = endpoint or os.getenv(
            "MINIMAX_TTS_ENDPOINT",
            "https://api.minimax.io/v1/t2a_v2",
        )
        self.endpoint = _http_endpoint(configured_endpoint)
        configured_websocket_endpoint = (
            websocket_endpoint
            or os.getenv("MINIMAX_TTS_WEBSOCKET_ENDPOINT")
            or _websocket_endpoint(configured_endpoint)
        )
        self.websocket_endpoint = _websocket_endpoint(configured_websocket_endpoint)
        self.language_boost = language_boost or os.getenv("MINIMAX_TTS_LANGUAGE_BOOST", "auto")
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.stream = _env_flag("MINIMAX_TTS_STREAM", True) if stream is None else stream
        configured_warmup_text = (
            model_warmup_text
            if model_warmup_text is not None
            else os.getenv("MINIMAX_TTS_MODEL_WARMUP_TEXT", _DEFAULT_MODEL_WARMUP_TEXT)
        )
        self.model_warmup_text = configured_warmup_text.strip() or _DEFAULT_MODEL_WARMUP_TEXT
        self.prime_texts = _parse_prime_texts(
            prime_texts if prime_texts is not None else os.getenv("MINIMAX_TTS_PRIME_TEXTS", "")
        )

        if not self.api_key:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_API_KEY.")
        if not self.voice_id:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_VOICE_ID.")

        resolved_keepalive_s = (
            _env_float("MINIMAX_TTS_CONNECTION_KEEPALIVE_S", _DEFAULT_KEEPALIVE_S)
            if connection_keepalive_s is None
            else float(connection_keepalive_s)
        )
        if resolved_keepalive_s <= 0:
            raise ValueError("MiniMax TTS connection_keepalive_s must be greater than zero.")
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(request_timeout_s, connect=5.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=8,
                keepalive_expiry=resolved_keepalive_s,
            ),
        )
        self._owns_client = client is None
        self._websocket_connect = websocket_connect
        self._owns_websocket_connector = websocket_connect is None
        self._websocket_open_timeout_s = (
            _env_float("MINIMAX_TTS_WEBSOCKET_OPEN_TIMEOUT_S", DEFAULT_OPEN_TIMEOUT_S)
            if websocket_open_timeout_s is None
            else float(websocket_open_timeout_s)
        )
        self._websocket_receive_timeout_s = (
            _env_float("MINIMAX_TTS_WEBSOCKET_RECEIVE_TIMEOUT_S", request_timeout_s)
            if websocket_receive_timeout_s is None
            else float(websocket_receive_timeout_s)
        )
        self._websocket_max_idle_s = (
            _env_float("MINIMAX_TTS_WEBSOCKET_MAX_IDLE_S", _WEBSOCKET_MAX_IDLE_S)
            if websocket_max_idle_s is None
            else float(websocket_max_idle_s)
        )
        if (
            self._websocket_open_timeout_s <= 0
            or self._websocket_receive_timeout_s <= 0
            or self._websocket_max_idle_s <= 0
        ):
            raise ValueError("MiniMax TTS WebSocket timeouts must be greater than zero.")
        self._websocket_session: MiniMaxWebSocketSession | None = None
        resolved_cache_mb = (
            _env_float("MINIMAX_TTS_CACHE_MAX_MB", _DEFAULT_CACHE_MB)
            if cache_max_mb is None
            else float(cache_max_mb)
        )
        if resolved_cache_mb < 0:
            raise ValueError("MINIMAX_TTS_CACHE_MAX_MB must be zero or greater.")
        self._cache_max_bytes = int(resolved_cache_mb * 1024 * 1024)
        self._audio_cache = _SHARED_AUDIO_CACHE if self._owns_client else _AudioLRUCache()
        self._connection_probe_lock = RLock()
        self._last_connection_use_s = 0.0
        self._last_model_use_s = 0.0

        default_warm = self._owns_client and (not self.stream or self._owns_websocket_connector)
        should_warm = (
            _env_flag("MINIMAX_TTS_WARMUP", default_warm)
            if warmup_connection is None
            else warmup_connection
        )
        self._connection_warmup_enabled = should_warm
        self._model_warmup_enabled = (
            _env_flag("MINIMAX_TTS_MODEL_WARMUP", default_warm)
            if warmup_model is None
            else warmup_model
        )
        if self._model_warmup_enabled:
            self._warmup_model()
        elif should_warm:
            self._warmup_connection()
        # Priming needs a live connection, so it follows warmup.
        self._prime_cache()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _warmup_connection(self) -> None:
        """Establish the streaming task, or prime HTTP for one-shot mode."""
        started_at_s = perf_counter()
        try:
            if self.stream:
                with self._connection_probe_lock:
                    self._ensure_websocket_session_locked()
                logger.info(
                    "MiniMax TTS WebSocket task ready in %.3fs",
                    perf_counter() - started_at_s,
                )
                return

            parts = urlsplit(self.endpoint)
            parent = parts.path.rsplit("/", 1)[0]
            endpoint = urlunsplit((parts.scheme, parts.netloc, f"{parent}/get_voice", "", ""))
            response = self.client.post(
                endpoint,
                headers=self._headers(),
                json={"voice_type": "system"},
                timeout=_WARMUP_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                self._raise_if_failed(body)
            self._last_connection_use_s = perf_counter()
            logger.info("MiniMax HTTP connection warmed in %.3fs", self._last_connection_use_s - started_at_s)
        except Exception as exc:
            logger.warning("MiniMax TTS connection warmup failed; continuing without it: %s", exc)

    def _warmup_model(self) -> None:
        """Run hidden synthesis and cache its PCM without sending it to playback."""
        started_at_s = perf_counter()
        try:
            if self.stream:
                chunks = [
                    self._chunk_bytes(chunk)
                    for chunk in self._iter_websocket_pcm(self.model_warmup_text, generation=None)
                ]
                self._cache_put(self._cache_key(self.model_warmup_text), chunks)
            else:
                response = self.client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=self._payload(self.model_warmup_text),
                    timeout=_WARMUP_TIMEOUT_S,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict):
                    self._raise_if_failed(body)
            self._last_connection_use_s = perf_counter()
            self._last_model_use_s = self._last_connection_use_s
            logger.info("MiniMax TTS model warmed in %.3fs", self._last_model_use_s - started_at_s)
        except Exception as exc:
            logger.warning("MiniMax TTS model warmup failed; continuing without it: %s", exc)

    def _prime_cache(self) -> None:
        """Pre-synthesise stock opening clauses into the exact-text cache.

        The clause-early first flush sends a reply's opening clause to TTS as its
        own request, so on a telephony call the very first thing the caller hears
        is almost always one of a handful of short acknowledgements. Synthesising
        those once at startup turns that request into a cache hit and takes the
        provider's first-byte time (~200ms on the measured profile) off the front
        of every turn that opens with one.

        Each entry costs one billable synthesis at startup, so the list is empty
        by default. Entries must match the flushed chunk exactly, punctuation
        included ("好的，", not "好的").
        """
        if not self.prime_texts:
            return
        started_at_s = perf_counter()
        primed = 0
        for text in self.prime_texts:
            key = self._cache_key(text)
            if self._cache_get(key) is not None:
                continue
            try:
                if self.stream:
                    chunks = [
                        self._chunk_bytes(chunk) for chunk in self._iter_websocket_pcm(text, generation=None)
                    ]
                else:
                    response = self.client.post(
                        self.endpoint,
                        headers=self._headers(),
                        json=self._payload(text),
                        timeout=_WARMUP_TIMEOUT_S,
                    )
                    response.raise_for_status()
                    body = response.json()
                    self._raise_if_failed(body)
                    data = body.get("data") or {}
                    if not data.get("audio"):
                        continue
                    chunks = [
                        self._chunk_bytes(chunk)
                        for chunk in self._emit_pcm(self._decode_wav(data["audio"]), None, pad=True)
                    ]
            except Exception as exc:
                # A prime failure only costs the speed-up, never the turn.
                logger.warning("MiniMax TTS cache priming failed for %r; continuing: %s", text, exc)
                continue
            if chunks:
                self._cache_put(key, chunks)
                primed += 1
        if primed:
            self._last_connection_use_s = perf_counter()
            self._last_model_use_s = self._last_connection_use_s
            logger.info(
                "MiniMax TTS primed %d/%d opening clauses in %.3fs",
                primed,
                len(self.prime_texts),
                perf_counter() - started_at_s,
            )

    def prewarm(self) -> None:
        """Refresh idle MiniMax connectivity/model state before a spoken turn."""
        if not self._connection_warmup_enabled and not self._model_warmup_enabled:
            return
        now = perf_counter()
        connection_stale = self._connection_is_stale(now)
        model_stale = (
            self._model_warmup_enabled
            and now - self._last_model_use_s >= _CONNECTION_PROBE_INTERVAL_S
        )
        if not connection_stale and not model_stale:
            return
        if not self._connection_probe_lock.acquire(blocking=False):
            return
        try:
            now = perf_counter()
            connection_stale = self._connection_is_stale(now)
            if (
                self._model_warmup_enabled
                and now - self._last_model_use_s >= _CONNECTION_PROBE_INTERVAL_S
            ):
                if connection_stale and self.stream:
                    self._close_websocket_session_locked()
                self._warmup_model()
            elif connection_stale:
                if self.stream:
                    self._close_websocket_session_locked()
                self._warmup_connection()
        finally:
            self._connection_probe_lock.release()

    def _connection_is_stale(self, now: float) -> bool:
        if self.stream:
            return (
                self._websocket_session is None
                or not self._websocket_session.started
                or now - self._last_connection_use_s >= self._websocket_max_idle_s
            )
        return now - self._last_connection_use_s >= _CONNECTION_PROBE_INTERVAL_S

    def maintain_connection(self) -> None:
        """Keep an idle telephony lane ready without running billable synthesis."""
        if (
            not self.stream
            or (not self._connection_warmup_enabled and not self._model_warmup_enabled)
            or not self._connection_is_stale(perf_counter())
            or not self._connection_probe_lock.acquire(blocking=False)
        ):
            return
        try:
            if not self._connection_is_stale(perf_counter()):
                return
            self._close_websocket_session_locked()
            started_at_s = perf_counter()
            self._ensure_websocket_session_locked()
            logger.info(
                "MiniMax TTS idle WebSocket task refreshed in %.3fs",
                perf_counter() - started_at_s,
            )
        except Exception as exc:
            self._close_websocket_session_locked()
            logger.warning("MiniMax TTS idle WebSocket refresh failed: %s", exc)
        finally:
            self._connection_probe_lock.release()

    def _task_start_payload(self) -> dict[str, Any]:
        return {
            "event": "task_start",
            "model": self.model,
            "language_boost": self.language_boost,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": 1.0,
                "pitch": 0,
                "english_normalization": False,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "format": "pcm",
                "channel": 1,
            },
            # MiniMax documents false as the lower-latency segmentation mode.
            "continuous_sound": False,
        }

    def _ensure_websocket_session_locked(self) -> MiniMaxWebSocketSession:
        session = self._websocket_session
        if session is not None and session.started:
            return session
        session = MiniMaxWebSocketSession(
            endpoint=self.websocket_endpoint,
            api_key=self.api_key,
            task_start=self._task_start_payload(),
            connect=self._websocket_connect,
            open_timeout_s=self._websocket_open_timeout_s,
            receive_timeout_s=self._websocket_receive_timeout_s,
        )
        session.start()
        self._websocket_session = session
        self._last_connection_use_s = perf_counter()
        return session

    def _release_failed_session_locked(self, exc: Exception) -> None:
        """Decide whether a failed synthesis costs us the warm session.

        A provider rejection -- a rate limit above all -- says nothing about the
        socket: the connection is healthy and the task is very likely still
        usable. Closing it anyway means the next turn pays a fresh connect and
        task_start handshake, which is exactly the cold-start cost this profile
        works to avoid, and it bites hardest under sustained rate limiting when
        every turn would then pay it.

        task_cancel doubles as a liveness probe: a server that answers
        task_canceled has demonstrably kept the task, so the session is kept.
        Anything else falls back to closing.
        """
        if isinstance(exc, MiniMaxProviderError):
            session = self._websocket_session
            if session is not None and session.cancel(timeout_s=_FAILED_SESSION_PROBE_S):
                logger.info(
                    "MiniMax TTS rejected the request (status_code=%s); session kept warm",
                    getattr(exc, "status_code", "unknown"),
                )
                return
        self._close_websocket_session_locked()

    def _abort_websocket_session_locked(self) -> None:
        """End the current synthesis on a barge-in, keeping the socket if we can.

        The bidi protocol has task_cancel, which discards buffered text and
        returns the task to ``task_started``. That saves the reconnect and
        task_start handshake the next turn would otherwise pay -- on a phone call
        barge-ins are common, so this is the difference between an interruption
        costing nothing and costing a cold TTS connection.
        """
        session = self._websocket_session
        if session is not None and session.cancel():
            logger.debug("MiniMax TTS task cancelled; session kept warm")
            return
        self._close_websocket_session_locked()

    def _close_websocket_session_locked(self, *, graceful: bool = False) -> None:
        session, self._websocket_session = self._websocket_session, None
        if session is not None:
            session.close(graceful=graceful)

    def _cache_key(self, text: str) -> tuple[Any, ...]:
        return (
            self.websocket_endpoint if self.stream else self.endpoint,
            self.model,
            self.voice_id,
            self.speed,
            self.language_boost,
            self.sample_rate,
            self.blocksize,
            self.stream,
            text,
        )

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[bytes, ...] | None:
        return self._audio_cache.get(key)

    def _cache_put(self, key: tuple[Any, ...], chunks: list[bytes]) -> None:
        self._audio_cache.put(key, chunks, self._cache_max_bytes)

    @staticmethod
    def _chunk_bytes(chunk: bytes | np.ndarray) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        return np.asarray(chunk, dtype="<i2").tobytes()

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": text,
            "stream": False,
            "language_boost": self.language_boost,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "format": "wav",
                "channel": 1,
            },
        }

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    @staticmethod
    def _raise_if_failed(body: dict[str, Any]) -> None:
        base_response = body.get("base_resp") or {}
        status_code = base_response.get("status_code")
        if status_code not in (None, 0):
            raise RuntimeError(
                "MiniMax TTS request failed "
                f"(status_code={status_code!r}): {base_response.get('status_msg', 'unknown error')}"
            )

    def _decode_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
        except wave.Error as exc:
            raise ValueError("MiniMax returned an invalid WAV payload.") from exc

        if channels != 1:
            raise ValueError(f"MiniMax returned {channels} audio channels; expected mono.")
        if sample_width != 2:
            raise ValueError(f"MiniMax returned {sample_width * 8}-bit audio; expected signed PCM16.")
        if sample_rate != self.sample_rate:
            raise ValueError(f"MiniMax returned {sample_rate} Hz audio; expected {self.sample_rate} Hz.")
        return np.frombuffer(frames, dtype="<i2").copy()

    def _decode_wav(self, audio_hex: str) -> np.ndarray:
        try:
            wav_bytes = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        return self._decode_wav_bytes(wav_bytes)

    def _decode_audio_payload(self, audio_hex: str) -> np.ndarray:
        try:
            raw = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        if raw.startswith(b"RIFF"):
            return self._decode_wav_bytes(raw)
        if len(raw) % 2:
            raw = raw[:-1]
        if not raw:
            return np.array([], dtype=np.int16)
        return np.frombuffer(raw, dtype="<i2").copy()

    def _decode_hex_pcm(self, audio_hex: str, leftover_hex: str = "") -> tuple[np.ndarray, str]:
        hex_str = leftover_hex + (audio_hex or "")
        if not hex_str:
            return np.array([], dtype=np.int16), ""
        if len(hex_str) % 2:
            leftover_hex = hex_str[-1]
            hex_str = hex_str[:-1]
        else:
            leftover_hex = ""
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        if len(raw) % 2:
            leftover_hex = f"{raw[-1]:02x}" + leftover_hex
            raw = raw[:-1]
        if not raw:
            return np.array([], dtype=np.int16), leftover_hex
        return np.frombuffer(raw, dtype="<i2").copy(), leftover_hex

    @staticmethod
    def _retryable_websocket_error(exc: Exception) -> bool:
        return isinstance(exc, (ConnectionClosed, ConnectionError, OSError, TimeoutError))

    def _iter_websocket_audio_hex(self, text: str) -> Iterator[str]:
        emitted_audio = False
        for attempt in range(2):
            with self._connection_probe_lock:
                try:
                    session = self._ensure_websocket_session_locked()
                    for audio_hex in session.synthesize(text):
                        emitted_audio = True
                        yield audio_hex
                    self._last_connection_use_s = perf_counter()
                    self._last_model_use_s = self._last_connection_use_s
                    return
                except Exception as exc:
                    self._release_failed_session_locked(exc)
                    if emitted_audio or attempt or not self._retryable_websocket_error(exc):
                        raise
                    logger.info("MiniMax TTS WebSocket was stale; reconnecting once")

    def _emit_pcm(self, samples: np.ndarray, generation: int | None, *, pad: bool) -> Iterator[np.ndarray]:
        if samples.size == 0:
            return
        for start in range(0, len(samples), self.blocksize):
            if self._is_cancelled(generation):
                logger.info("MiniMax TTS playback cancelled (interruption)")
                return
            chunk = samples[start : start + self.blocksize]
            if len(chunk) < self.blocksize:
                if not pad:
                    return
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield np.asarray(chunk, dtype=np.int16)

    def _log_first_audio_latency(self, tts_input: TTSInput, started_at_s: float) -> None:
        request_s = perf_counter() - started_at_s
        logger.info(
            "MiniMax TTS first audio in %.3fs (turn=%s rev=%s)",
            request_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )
        if tts_input.speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - tts_input.speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _iter_websocket_pcm(
        self,
        text: str,
        generation: int | None,
    ) -> Iterator[np.ndarray]:
        leftover_hex = ""
        pending = np.array([], dtype=np.int16)
        got_audio = False
        first_audio = True

        for audio_hex in self._iter_websocket_audio_hex(text):
            if self._is_cancelled(generation):
                with self._connection_probe_lock:
                    self._abort_websocket_session_locked()
                logger.info("MiniMax TTS playback cancelled (interruption)")
                return
            got_audio = True
            samples, leftover_hex = self._decode_hex_pcm(audio_hex, leftover_hex)
            if samples.size == 0:
                continue
            pending = np.concatenate((pending, samples)) if pending.size else samples
            if first_audio:
                first_audio = False
                # Don't add one playback-block delay to a short first provider frame.
                if len(pending) < self.blocksize:
                    yield np.asarray(pending, dtype=np.int16)
                    pending = np.array([], dtype=np.int16)
                    continue
            emit_upto = len(pending) - (len(pending) % self.blocksize)
            if emit_upto:
                yield from self._emit_pcm(pending[:emit_upto], generation, pad=False)
                pending = pending[emit_upto:]
                if self._is_cancelled(generation):
                    with self._connection_probe_lock:
                        self._abort_websocket_session_locked()
                    return

        if self._is_cancelled(generation):
            with self._connection_probe_lock:
                self._abort_websocket_session_locked()
            return
        if not got_audio:
            # Punctuation-only or pause-only text can complete with empty audio.
            logger.info(
                "MiniMax TTS returned no audio for %r; treating as silent",
                text if len(text) <= 80 else f"{text[:80]}…",
            )
            return
        if leftover_hex:
            raise ValueError("MiniMax TTS WebSocket returned incomplete hex audio.")
        yield from self._emit_pcm(pending, generation, pad=True)

    def _synthesize_streaming(
        self,
        text: str,
        generation: int | None,
        tts_input: TTSInput,
    ) -> Iterator[np.ndarray]:
        started_at_s = perf_counter()
        first_audio = True
        for chunk in self._iter_websocket_pcm(text, generation):
            if first_audio:
                self._log_first_audio_latency(tts_input, started_at_s)
                first_audio = False
            yield chunk

    def _synthesize_sync(
        self,
        text: str,
        generation: int | None,
        tts_input: TTSInput,
    ) -> Iterator[np.ndarray]:
        started_at_s = perf_counter()
        response = self.client.post(
            self.endpoint,
            headers=self._headers(),
            json=self._payload(text),
        )
        response.raise_for_status()
        self._last_connection_use_s = perf_counter()
        self._last_model_use_s = self._last_connection_use_s
        body = response.json()
        self._raise_if_failed(body)

        data = body.get("data")
        if not data or not data.get("audio"):
            raise RuntimeError("MiniMax TTS response did not contain audio data.")
        audio = self._decode_wav(data["audio"])
        if audio.size:
            self._log_first_audio_latency(tts_input, started_at_s)
        yield from self._emit_pcm(audio, generation, pad=True)

    def _synthesize_and_cache(
        self,
        text: str,
        generation: int | None,
        tts_input: TTSInput,
        key: tuple[Any, ...],
    ) -> Iterator[TTSOut]:
        producer = (
            self._synthesize_streaming(text, generation, tts_input)
            if self.stream
            else self._synthesize_sync(text, generation, tts_input)
        )
        if self._cache_max_bytes <= 0:
            yield from producer
            return

        chunks: list[bytes] = []
        for chunk in producer:
            chunks.append(self._chunk_bytes(chunk))
            yield chunk
        if chunks and not self._is_cancelled(generation):
            self._cache_put(key, chunks)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale MiniMax TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        generation = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text.strip()
        if not text:
            return
        if not _has_speakable_content(text):
            logger.info("Skipping MiniMax TTS for non-speech text: %r", text)
            return
        cache_key = self._cache_key(text)
        cached = self._cache_get(cache_key)
        console.print(f"[green]ASSISTANT: {text}")
        logger.info(
            "Streaming TTS start (%s, turn=%s rev=%s): %s",
            "cache" if cached is not None else ("websocket" if self.stream else "oneshot"),
            tts_input.turn_id,
            tts_input.turn_revision,
            text if len(text) <= 80 else f"{text[:80]}…",
        )

        if cached is not None:
            started_at_s = perf_counter()
            logger.info("MiniMax TTS cache hit (%d chunks)", len(cached))
            if cached:
                self._log_first_audio_latency(tts_input, started_at_s)
            for chunk in cached:
                if self._is_cancelled(generation):
                    logger.info("MiniMax TTS cached playback cancelled (interruption)")
                    return
                yield chunk
        else:
            yield from self._synthesize_and_cache(text, generation, tts_input, cache_key)
        logger.info(
            "Streaming TTS finished (turn=%s rev=%s)",
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def on_session_end(self) -> None:
        with self._connection_probe_lock:
            self._close_websocket_session_locked(graceful=True)

    def cleanup(self) -> None:
        with self._connection_probe_lock:
            self._close_websocket_session_locked(graceful=True)
        if self._owns_client:
            self.client.close()
