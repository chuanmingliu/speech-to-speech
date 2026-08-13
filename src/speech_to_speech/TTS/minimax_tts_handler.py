from __future__ import annotations

import json
import logging
import os
import ssl
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Callable, Iterator

import numpy as np
from websockets.sync.client import connect

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.incremental_mp3_decoder import IncrementalMP3Decoder

logger = logging.getLogger(__name__)

_OFFICIAL_MINIMAX_STREAMING_ENDPOINTS = frozenset(
    {
        "wss://api.minimax.io/ws/v1/t2a_v2",
        "wss://api.minimaxi.com/ws/v1/t2a_v2",
    }
)


def _silent_dependency_logger() -> logging.Logger:
    dependency_logger = logging.Logger("speech_to_speech.minimax.private_websocket", level=logging.CRITICAL + 1)
    dependency_logger.disabled = True
    dependency_logger.propagate = False
    dependency_logger.addHandler(logging.NullHandler())
    return dependency_logger


@dataclass(frozen=True)
class MiniMaxTTSConfig:
    api_key: str
    voice_id: str
    model: str = "speech-2.8-turbo"
    endpoint: str = "wss://api.minimax.io/ws/v1/t2a_v2"
    language_boost: str = "auto"
    sample_rate: int = 16_000
    channels: int = 1
    block_samples: int = 512
    bitrate: int = 128_000
    open_timeout_s: float = 10.0
    close_timeout_s: float = 5.0
    read_poll_timeout_s: float = 0.1
    write_timeout_s: float = 0.1
    event_timeout_s: float = 30.0
    max_event_bytes: int = 1024 * 1024
    max_audio_bytes: int = 512 * 1024
    max_queue: int = 4


class _MiniMaxCancelled(Exception):
    pass


class _OperationState:
    def __init__(self) -> None:
        self.done = Event()
        self.lock = Lock()
        self.abandoned = False
        self.result: Any = None
        self.error: BaseException | None = None


class MiniMaxStreamingClient:
    """One official MiniMax WebSocket T2A task."""

    def __init__(
        self,
        config: MiniMaxTTSConfig,
        *,
        connect_fn: Callable[..., Any] = connect,
        decoder_factory: Callable[..., Any] = IncrementalMP3Decoder,
    ) -> None:
        if config.endpoint not in _OFFICIAL_MINIMAX_STREAMING_ENDPOINTS:
            raise ValueError("MiniMax streaming endpoint must be the official secure WebSocket endpoint")
        if not config.api_key or not config.voice_id:
            raise ValueError("MiniMax API key and voice ID are required")
        if config.sample_rate != 16_000 or config.channels != 1 or config.block_samples != 512:
            raise ValueError("MiniMax output must be 16 kHz mono with 512-sample blocks")
        self.config = config
        self._connect = connect_fn
        self._decoder = decoder_factory(
            sample_rate=config.sample_rate,
            channels=config.channels,
            block_samples=config.block_samples,
        )
        self._websocket: Any | None = None
        self._started = False
        self._finished = False
        self._closed = False
        self._operation_guard = Lock()
        self._operation_abandoned = False
        self._close_requested = Event()

    def start(self, *, cancelled: Callable[[], bool] = lambda: False) -> None:
        if self._closed:
            raise RuntimeError("MiniMax client is closed")
        if self._started:
            return
        tls = ssl.create_default_context()
        try:
            self._websocket = self._run_bounded_operation(
                lambda: self._connect(
                    self.config.endpoint,
                    ssl=tls,
                    additional_headers={"Authorization": f"Bearer {self.config.api_key}"},
                    proxy=None,
                    compression=None,
                    max_size=self.config.max_event_bytes,
                    max_queue=self.config.max_queue,
                    open_timeout=self.config.open_timeout_s,
                    close_timeout=self.config.close_timeout_s,
                    logger=_silent_dependency_logger(),
                ),
                timeout_s=self.config.open_timeout_s,
                cancelled=cancelled,
                operation_name="connection",
                late_result_cleanup=self._abort_connection,
            )
            self._expect_event("connected_success", cancelled=cancelled)
            self._send(
                {
                    "event": "task_start",
                    "model": self.config.model,
                    "language_boost": self.config.language_boost,
                    "voice_setting": {
                        "voice_id": self.config.voice_id,
                        "speed": 1.1,
                        "vol": 1.0,
                        "pitch": 0,
                    },
                    "audio_setting": {
                        "sample_rate": self.config.sample_rate,
                        "bitrate": self.config.bitrate,
                        "format": "mp3",
                        "channel": self.config.channels,
                    },
                },
                cancelled=cancelled,
            )
            self._expect_event("task_started", cancelled=cancelled)
            if cancelled():
                raise _MiniMaxCancelled
            self._started = True
        except _MiniMaxCancelled:
            self._abort()
            return
        except Exception:
            self._abort()
            raise

    def synthesize(self, text: str, *, cancelled: Callable[[], bool]) -> Iterator[np.ndarray]:
        if not self._started or self._finished or self._closed:
            raise RuntimeError("MiniMax task is not active")
        if cancelled():
            self._abort()
            return
        try:
            self._send({"event": "task_continue", "text": text}, cancelled=cancelled)
            while True:
                event = self._receive_event(cancelled=cancelled)
                event_name = event.get("event")
                if event_name not in (None, "task_continued"):
                    raise RuntimeError("MiniMax returned an event out of order")
                data = event.get("data")
                if data is not None and not isinstance(data, dict):
                    raise RuntimeError("MiniMax returned malformed audio data")
                audio_hex = data.get("audio") if data else None
                if audio_hex is not None:
                    if not isinstance(audio_hex, str) or len(audio_hex) % 2:
                        raise RuntimeError("MiniMax returned malformed hex audio")
                    if len(audio_hex) // 2 > self.config.max_audio_bytes:
                        raise RuntimeError("MiniMax audio fragment exceeds its bound")
                    try:
                        encoded = bytes.fromhex(audio_hex)
                    except ValueError as exc:
                        raise RuntimeError("MiniMax returned malformed hex audio") from exc
                    for block in self._decoder.feed(encoded):
                        if cancelled():
                            raise _MiniMaxCancelled
                        yield block
                is_final = event.get("is_final")
                if not isinstance(is_final, bool):
                    raise RuntimeError("MiniMax response is missing a terminal marker")
                if is_final:
                    return
                if cancelled():
                    raise _MiniMaxCancelled
        except _MiniMaxCancelled:
            self._abort()
            return
        except Exception:
            self._abort()
            raise

    def finish(self, *, cancelled: Callable[[], bool] = lambda: False) -> list[np.ndarray]:
        if self._closed:
            return []
        if not self._started:
            raise RuntimeError("MiniMax task was not started")
        if self._finished:
            return []
        try:
            self._send({"event": "task_finish"}, cancelled=cancelled)
            self._expect_event("task_finished", cancelled=cancelled)
            if cancelled():
                raise _MiniMaxCancelled
            blocks = self._decoder.finish()
            if cancelled():
                raise _MiniMaxCancelled
            self._finished = True
            return blocks
        except _MiniMaxCancelled:
            self._abort()
            return []
        except Exception:
            self._abort()
            raise

    def close(self, *, graceful: bool = False) -> None:
        if self._closed:
            return
        self._close_requested.set()
        self._closed = True
        self._decoder.close()
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            if graceful:
                try:
                    websocket.close()
                except Exception:
                    self._abort_connection(websocket)
            else:
                self._abort_connection(websocket)

    def _abort(self) -> None:
        if self._closed:
            return
        self._close_requested.set()
        self._closed = True
        self._decoder.close()
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            self._abort_connection(websocket)

    def _send(self, message: dict[str, Any], *, cancelled: Callable[[], bool]) -> None:
        if self._websocket is None:
            raise RuntimeError("MiniMax WebSocket is not connected")
        if cancelled():
            raise _MiniMaxCancelled
        websocket = self._websocket
        payload = json.dumps(message, separators=(",", ":"))
        self._run_bounded_operation(
            lambda: websocket.send(payload),
            timeout_s=self.config.write_timeout_s,
            cancelled=cancelled,
            operation_name="write",
        )
        if cancelled():
            raise _MiniMaxCancelled

    def _run_bounded_operation(
        self,
        operation: Callable[[], Any],
        *,
        timeout_s: float,
        cancelled: Callable[[], bool],
        operation_name: str,
        late_result_cleanup: Callable[[Any], None] | None = None,
    ) -> Any:
        if timeout_s <= 0:
            raise ValueError(f"{operation_name} timeout must be positive")
        if self._operation_abandoned:
            raise RuntimeError("MiniMax client has an abandoned operation")
        if not self._operation_guard.acquire(blocking=False):
            raise RuntimeError("MiniMax client already has an operation in progress")
        if self._operation_abandoned:
            self._operation_guard.release()
            raise RuntimeError("MiniMax client has an abandoned operation")
        state = _OperationState()

        def run() -> None:
            try:
                result = operation()
            except BaseException as exc:
                with state.lock:
                    if state.abandoned:
                        return
                    state.error = exc
                    state.done.set()
                return
            cleanup = False
            with state.lock:
                if state.abandoned:
                    cleanup = late_result_cleanup is not None
                else:
                    state.result = result
                    state.done.set()
            if cleanup and late_result_cleanup is not None:
                late_result_cleanup(result)

        worker = Thread(target=run, name=f"minimax-{operation_name}", daemon=True)
        worker.start()
        deadline = monotonic() + timeout_s
        try:
            while True:
                if self._close_requested.is_set() or cancelled():
                    self._operation_abandoned = True
                    self._abandon_operation(state, late_result_cleanup)
                    raise _MiniMaxCancelled
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._operation_abandoned = True
                    self._abandon_operation(state, late_result_cleanup)
                    raise TimeoutError(f"MiniMax {operation_name} timed out")
                if state.done.wait(min(self.config.read_poll_timeout_s, remaining)):
                    if self._close_requested.is_set() or cancelled():
                        self._operation_abandoned = True
                        self._abandon_operation(state, late_result_cleanup)
                        raise _MiniMaxCancelled
                    break
            if state.error is not None:
                raise state.error
            return state.result
        finally:
            self._operation_guard.release()

    @staticmethod
    def _abandon_operation(
        state: _OperationState,
        late_result_cleanup: Callable[[Any], None] | None,
    ) -> None:
        result: Any = None
        cleanup = False
        with state.lock:
            state.abandoned = True
            if state.done.is_set() and state.error is None and late_result_cleanup is not None:
                result = state.result
                cleanup = True
        if cleanup:
            late_result_cleanup(result)

    @staticmethod
    def _abort_connection(websocket: Any) -> None:
        close_socket = getattr(websocket, "close_socket", None)
        if callable(close_socket):
            try:
                close_socket()
                return
            except Exception:
                pass
        transport = getattr(websocket, "socket", None)
        if transport is not None:
            try:
                transport.shutdown(2)
            except Exception:
                pass
            try:
                transport.close()
            except Exception:
                pass

    def _expect_event(self, expected: str, *, cancelled: Callable[[], bool]) -> dict[str, Any]:
        event = self._receive_event(cancelled=cancelled)
        if event.get("event") != expected:
            raise RuntimeError("MiniMax returned an event out of order")
        return event

    def _receive_event(self, *, cancelled: Callable[[], bool]) -> dict[str, Any]:
        if self._websocket is None:
            raise RuntimeError("MiniMax WebSocket is not connected")
        deadline = monotonic() + self.config.event_timeout_s
        while True:
            if cancelled():
                raise _MiniMaxCancelled
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("MiniMax provider event timed out")
            try:
                raw = self._websocket.recv(timeout=min(self.config.read_poll_timeout_s, remaining))
                break
            except TimeoutError:
                continue
        if not isinstance(raw, str):
            raise RuntimeError("MiniMax returned a non-text event")
        if len(raw.encode("utf-8")) > self.config.max_event_bytes:
            raise RuntimeError("MiniMax provider event exceeds its bound")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MiniMax returned malformed JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("MiniMax returned a non-object event")
        base_response = event.get("base_resp")
        if not isinstance(base_response, dict) or not isinstance(base_response.get("status_code"), int):
            raise RuntimeError("MiniMax event is missing provider status")
        status_code = base_response["status_code"]
        if status_code != 0 or event.get("event") == "task_failed":
            raise RuntimeError(f"MiniMax provider failed with status code {status_code}")
        return event


class MiniMaxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Own one MiniMax streaming task for each assistant response."""

    def setup(
        self,
        should_listen: Event,
        api_key: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        endpoint: str | None = None,
        language_boost: str | None = None,
        sample_rate: int = 16000,
        blocksize: int = 512,
        request_timeout_s: float = 30.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        client_factory: Callable[[MiniMaxTTSConfig], MiniMaxStreamingClient] = MiniMaxStreamingClient,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        resolved_api_key = api_key or os.getenv("MINIMAX_TTS_API_KEY")
        resolved_voice_id = voice_id or os.getenv("MINIMAX_TTS_VOICE_ID")
        if not resolved_api_key:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_API_KEY.")
        if not resolved_voice_id:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_VOICE_ID.")
        self._config = MiniMaxTTSConfig(
            api_key=resolved_api_key,
            voice_id=resolved_voice_id,
            model=model or os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-turbo"),
            endpoint=endpoint
            or os.getenv("MINIMAX_TTS_ENDPOINT", "wss://api.minimax.io/ws/v1/t2a_v2"),
            language_boost=language_boost or os.getenv("MINIMAX_TTS_LANGUAGE_BOOST", "auto"),
            sample_rate=sample_rate,
            block_samples=blocksize,
            event_timeout_s=request_timeout_s,
        )
        self._client_factory = client_factory
        self._clock = clock
        self._active_client: MiniMaxStreamingClient | None = None
        self._active_generation: int | None = None
        self._active_turn: tuple[str | None, int | None] | None = None
        self._request_started_at_s: float | None = None
        self._first_audio_at_s: float | None = None
        self._last_audio_at_s: float | None = None
        self._pending_first_audio_at_s: float | None = None
        self._phrase_dispatched = False

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                self._close_active_client_for((tts_input.turn_id, tts_input.turn_revision))
                return
            client = self._active_client
            if client is not None:
                try:
                    if self._active_cancelled():
                        self._close_active_client()
                    else:
                        for block in client.finish(cancelled=self._active_cancelled):
                            yield block
                        self._close_active_client(graceful=True)
                except Exception:
                    self._close_active_client()
                    raise
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            self._close_active_client_for((tts_input.turn_id, tts_input.turn_revision))
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

        turn = (tts_input.turn_id, tts_input.turn_revision)
        if self._active_client is not None and self._active_turn != turn:
            self._close_active_client()
        if self._active_cancelled():
            self._close_active_client()
            return
        if self._active_client is None:
            client = self._client_factory(self._config)
            self._active_client = client
            self._active_generation = generation
            self._active_turn = turn
            self._request_started_at_s = self._clock()
            self._first_audio_at_s = None
            self._last_audio_at_s = None
            self._phrase_dispatched = False
            try:
                client.start(cancelled=self._active_cancelled)
            except Exception:
                self._close_active_client()
                raise
            if self._active_client is not client:
                return
            if self._active_cancelled():
                self._close_active_client()
                return

        client = self._active_client
        assert client is not None
        try:
            dispatched_at_s = self._clock()
            dispatch_log = logger.debug if self._phrase_dispatched else logger.info
            self._phrase_dispatched = True
            if tts_input.speakable_phrase_at_s is not None:
                dispatch_log(
                    "MiniMax phrase-ready to dispatch latency: %.3fs (turn=%s rev=%s)",
                    dispatched_at_s - tts_input.speakable_phrase_at_s,
                    tts_input.turn_id,
                    tts_input.turn_revision,
                )
            for block in client.synthesize(text, cancelled=self._active_cancelled):
                if self._first_audio_at_s is None:
                    self._first_audio_at_s = self._clock()
                    self._last_audio_at_s = self._first_audio_at_s
                    self._pending_first_audio_at_s = self._first_audio_at_s
                    if self._request_started_at_s is not None:
                        logger.info(
                            "MiniMax request to first audio latency: %.3fs (turn=%s rev=%s)",
                            self._first_audio_at_s - self._request_started_at_s,
                            tts_input.turn_id,
                            tts_input.turn_revision,
                        )
                    if tts_input.speech_stopped_at_s is not None:
                        logger.info(
                            "Speech end to first audio latency: %.3fs (turn=%s rev=%s)",
                            self._first_audio_at_s - tts_input.speech_stopped_at_s,
                            tts_input.turn_id,
                            tts_input.turn_revision,
                        )
                else:
                    self._last_audio_at_s = self._clock()
                yield block
        except Exception:
            self._close_active_client()
            raise
        if self._active_cancelled():
            self._close_active_client()

    def _active_cancelled(self) -> bool:
        return self.stop_event.is_set() or (
            self._active_generation is not None
            and self.cancel_scope is not None
            and self.cancel_scope.is_stale(self._active_generation)
        )

    def output_for_queue(self, output: TTSOut, source_input: TTSIn) -> TTSOut | AudioOutput:
        queued = super().output_for_queue(output, source_input)
        if isinstance(queued, AudioOutput) and self._pending_first_audio_at_s is not None:
            queued.first_audio_at_s = self._pending_first_audio_at_s
            self._pending_first_audio_at_s = None
        return queued

    def _close_active_client(self, *, graceful: bool = False) -> None:
        client, self._active_client = self._active_client, None
        turn = self._active_turn
        cancelled_at_s = (
            self.cancel_scope.cancelled_at_s
            if self._active_generation is not None
            and self.cancel_scope is not None
            and self.cancel_scope.is_stale(self._active_generation)
            else None
        )
        self._active_generation = None
        self._active_turn = None
        self._request_started_at_s = None
        self._first_audio_at_s = None
        self._pending_first_audio_at_s = None
        self._phrase_dispatched = False
        if client is not None:
            client.close(graceful=graceful)
            if cancelled_at_s is not None:
                logger.info(
                    "MiniMax barge-in close latency: %.3fs (turn=%s rev=%s)",
                    self._clock() - cancelled_at_s,
                    turn[0] if turn else None,
                    turn[1] if turn else None,
                )
                if self._last_audio_at_s is not None:
                    logger.info(
                        "MiniMax last accepted audio offset from barge-in: %.3fs (turn=%s rev=%s)",
                        self._last_audio_at_s - cancelled_at_s,
                        turn[0] if turn else None,
                        turn[1] if turn else None,
                    )
        self._last_audio_at_s = None

    def _close_active_client_for(self, turn: tuple[str | None, int | None]) -> None:
        if self._active_turn == turn:
            self._close_active_client()

    def on_session_end(self) -> None:
        self._close_active_client()

    def cleanup(self) -> None:
        self._close_active_client()
