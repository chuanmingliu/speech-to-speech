from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WEBSOCKET_ENDPOINT = "wss://api.minimax.io/ws/v1/t2a_v2"
DEFAULT_OPEN_TIMEOUT_S = 5.0
DEFAULT_RECEIVE_TIMEOUT_S = 30.0


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
    """Persistent MiniMax T2A task following the official WebSocket protocol."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        task_start: dict[str, Any],
        connect: Callable[..., Any] | None = None,
        open_timeout_s: float = DEFAULT_OPEN_TIMEOUT_S,
        receive_timeout_s: float = DEFAULT_RECEIVE_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.task_start = task_start
        self.open_timeout_s = open_timeout_s
        self.receive_timeout_s = receive_timeout_s
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
        while True:
            message = self._recv(phase="synthesis", timeout_s=self.receive_timeout_s)
            self._raise_if_failed(message)
            event = message.get("event")
            if event == "task_failed":
                raise RuntimeError("MiniMax TTS WebSocket task failed.")
            if event not in (None, "task_continued"):
                raise RuntimeError(f"MiniMax TTS WebSocket returned unexpected event {event!r}.")

            data = message.get("data") or {}
            audio = data.get("audio")
            if audio:
                if not isinstance(audio, str):
                    raise ValueError("MiniMax TTS WebSocket audio must be hex text.")
                yield audio
            if message.get("is_final") is True:
                return

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
            raise RuntimeError(
                "MiniMax TTS request failed "
                f"(status_code={status_code!r}): {base_response.get('status_msg', 'unknown error')}"
            )
