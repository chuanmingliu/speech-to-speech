from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# The bidirectional endpoint. Unlike ``/ws/v1/t2a_v2`` it assembles sentences
# server-side, emits sentence_start/sentence_end, and supports task_cancel and
# task_flush without tearing the connection down.
DEFAULT_WEBSOCKET_ENDPOINT = "wss://api.minimax.io/ws/v1/t2a_v2_bidi"
DEFAULT_OPEN_TIMEOUT_S = 5.0
DEFAULT_RECEIVE_TIMEOUT_S = 30.0
DEFAULT_CANCEL_TIMEOUT_S = 1.0

# Informational frames the bidi server interleaves with audio. They carry no
# audio of their own and must not end a synthesis.
_BIDI_PROGRESS_EVENTS = frozenset({"sentence_start", "sentence_end", "task_continued"})


class MiniMaxProviderError(RuntimeError):
    """The provider rejected a request at the application layer.

    Distinct from a transport failure: the socket is healthy and the task may
    still be usable, so the caller can decide whether the warm session is worth
    keeping. A rate-limit rejection (status_code 1002) is the common case.
    """

    def __init__(self, message: str, *, status_code: object = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_bidi_endpoint(endpoint: str) -> bool:
    """Whether ``endpoint`` addresses the bidirectional T2A protocol."""
    return urlsplit(endpoint).path.rstrip("/").endswith("_bidi")


def _default_connect(
    endpoint: str,
    *,
    headers: dict[str, str],
    open_timeout_s: float,
) -> Any:
    from websockets.sync.client import connect

    return connect(
        endpoint,
        additional_headers=headers,
        open_timeout=open_timeout_s,
        close_timeout=1.0,
        ping_interval=20.0,
        ping_timeout=20.0,
    )


class MiniMaxWebSocketSession:
    """Persistent MiniMax T2A task.

    Speaks the bidirectional protocol (``/ws/v1/t2a_v2_bidi``) when the endpoint
    names it, and the original ``/ws/v1/t2a_v2`` protocol otherwise. The two
    differ in ways that matter here:

    * The bidi server buffers ``task_continue`` text until *it* judges a sentence
      complete. A short opening clause ("好的，") would therefore sit
      unsynthesised, which is exactly the chunk this pipeline most wants back
      quickly, so every synthesis is followed by ``task_flush`` to force it out.
    * Synthesis ends on ``task_flushed`` rather than an ``is_final`` audio frame,
      and ``sentence_start`` / ``sentence_end`` are interleaved.
    * ``task_cancel`` discards buffered text and returns the session to
      ``task_started``, so a barge-in no longer costs a reconnect and handshake.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        task_start: dict[str, Any],
        connect: Callable[..., Any] | None = None,
        open_timeout_s: float = DEFAULT_OPEN_TIMEOUT_S,
        receive_timeout_s: float = DEFAULT_RECEIVE_TIMEOUT_S,
        cancel_timeout_s: float = DEFAULT_CANCEL_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.task_start = task_start
        self.open_timeout_s = open_timeout_s
        self.receive_timeout_s = receive_timeout_s
        self.cancel_timeout_s = cancel_timeout_s
        self.bidi = is_bidi_endpoint(endpoint)
        self._connect = connect or _default_connect
        self._ws: Any | None = None
        self.started = False

    def start(self) -> None:
        self.close()
        try:
            self._ws = self._connect(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                open_timeout_s=self.open_timeout_s,
            )
            self._expect("connected_success", phase="connection")
            self._send(self.task_start)
            self._expect("task_started", phase="task start")
            self.started = True
        except Exception:
            self.close()
            raise

    def synthesize(self, text: str) -> Iterator[str]:
        if self._ws is None or not self.started:
            raise RuntimeError("MiniMax TTS WebSocket session is not started.")

        self._send({"event": "task_continue", "text": text})
        if self.bidi:
            # Without this the server may hold a short clause in its sentence
            # buffer indefinitely, waiting for punctuation that never comes.
            self._send({"event": "task_flush"})

        while True:
            message = self._recv(phase="synthesis", timeout_s=self.receive_timeout_s)
            self._raise_if_failed(message)
            event = message.get("event")
            if event == "task_failed":
                # A failed task cannot be reused, so this is not recoverable
                # the way a plain application rejection is.
                raise RuntimeError("MiniMax TTS WebSocket task failed.")

            data = message.get("data") or {}
            audio = data.get("audio")
            if audio:
                if not isinstance(audio, str):
                    raise ValueError("MiniMax TTS WebSocket audio must be hex text.")
                yield audio

            if self.bidi:
                if event == "task_flushed":
                    return
                if event is not None and event not in _BIDI_PROGRESS_EVENTS:
                    # Tolerate frames this client does not model rather than
                    # failing a live turn; the receive timeout still bounds us.
                    logger.debug("Ignoring MiniMax bidi event %r", event)
                continue

            if event not in (None, "task_continued"):
                raise RuntimeError(f"MiniMax TTS WebSocket returned unexpected event {event!r}.")
            if message.get("is_final") is True:
                return

    def cancel(self, timeout_s: float | None = None) -> bool:
        """Discard buffered text, keeping the session open.

        Returns ``True`` when the session survived and can synthesise again.
        ``False`` means the caller must close it — the original protocol has no
        interrupt, so it always returns ``False`` there.
        """
        if not self.bidi or self._ws is None or not self.started:
            return False
        try:
            self._send({"event": "task_cancel"})
            deadline = time.monotonic() + (self.cancel_timeout_s if timeout_s is None else timeout_s)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.info("MiniMax TTS task_cancel timed out; dropping the session")
                    return False
                message = self._recv(phase="cancel", timeout_s=remaining)
                self._raise_if_failed(message)
                event = message.get("event")
                if event == "task_canceled":
                    return True
                if event == "task_failed":
                    return False
                # Audio already in flight arrives before the acknowledgement; the
                # caller has stopped consuming, so it is simply drained here.
        except Exception:
            logger.debug("MiniMax TTS task_cancel failed", exc_info=True)
            return False

    def close(self, *, graceful: bool = False) -> None:
        ws, self._ws = self._ws, None
        was_started, self.started = self.started, False
        if ws is None:
            return
        if graceful and was_started:
            try:
                ws.send(json.dumps({"event": "task_finish"}))
            except Exception:
                logger.debug("MiniMax TTS task_finish failed", exc_info=True)
        try:
            ws.close()
        except Exception:
            logger.debug("MiniMax TTS WebSocket close failed", exc_info=True)

    def _send(self, message: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("MiniMax TTS WebSocket is not connected.")
        self._ws.send(json.dumps(message))

    def _recv(self, *, phase: str, timeout_s: float) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("MiniMax TTS WebSocket is not connected.")
        try:
            raw = self._ws.recv(timeout=timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"MiniMax TTS WebSocket {phase} timed out.") from exc
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            message = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("MiniMax TTS WebSocket returned invalid JSON.") from exc
        if not isinstance(message, dict):
            raise ValueError("MiniMax TTS WebSocket event must be a JSON object.")
        return message

    def _expect(self, expected: str, *, phase: str) -> dict[str, Any]:
        message = self._recv(phase=phase, timeout_s=self.open_timeout_s)
        self._raise_if_failed(message)
        actual = message.get("event")
        if actual != expected:
            raise RuntimeError(
                f"MiniMax TTS WebSocket {phase} returned {actual!r}; expected {expected!r}."
            )
        return message

    @staticmethod
    def _raise_if_failed(message: dict[str, Any]) -> None:
        base_response = message.get("base_resp") or {}
        status_code = base_response.get("status_code")
        if status_code not in (None, 0):
            raise MiniMaxProviderError(
                "MiniMax TTS request failed "
                f"(status_code={status_code!r}): {base_response.get('status_msg', 'unknown error')}",
                status_code=status_code,
            )
