from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

REALTIME_HOST_PATH = "asr.cloud.tencent.com/asr/v2"
PCM_VOICE_FORMAT = 1
# 200 ms of 16 kHz mono PCM16.
PCM_FRAME_BYTES = 6400


def build_realtime_url(
    *,
    app_id: str,
    secret_id: str,
    secret_key: str,
    engine: str,
    voice_id: str | None = None,
    timestamp: int | None = None,
    nonce: int | None = None,
    expired: int | None = None,
) -> str:
    """Build a signed Tencent realtime ASR WebSocket URL.

    The signature is HMAC-SHA1 over the unquoted host/path/query, matching the
    official speech SDK. Values are URL-encoded only in the request URL.
    """
    now = int(timestamp if timestamp is not None else time.time())
    voice_id = voice_id or str(uuid.uuid4())
    params: dict[str, str] = {
        "convert_num_mode": "0",
        "engine_model_type": engine,
        "expired": str(expired if expired is not None else now + 24 * 60 * 60),
        "filter_dirty": "0",
        "filter_modal": "0",
        "filter_punc": "0",
        "needvad": "0",
        "nonce": str(nonce if nonce is not None else now),
        "secretid": secret_id,
        "sub_service_type": "1",
        "timestamp": str(now),
        "voice_format": str(PCM_VOICE_FORMAT),
        "voice_id": voice_id,
        "word_info": "0",
    }
    sign_src = f"{REALTIME_HOST_PATH}/{app_id}?" + "&".join(f"{key}={params[key]}" for key in sorted(params))
    digest = hmac.new(secret_key.encode("utf-8"), sign_src.encode("utf-8"), hashlib.sha1).digest()
    signature = base64.b64encode(digest).decode("ascii")
    query = "&".join(f"{key}={quote(params[key], safe='')}" for key in sorted(params))
    query += f"&signature={quote(signature, safe='')}"
    return f"wss://{REALTIME_HOST_PATH}/{app_id}?{query}"


def _default_connect(url: str) -> Any:
    from websockets.sync.client import connect

    return connect(url, open_timeout=5.0, close_timeout=2.0)


class TencentRealtimeASRSession:
    """One WebSocket recognition stream (one voice_id)."""

    def __init__(
        self,
        url: str,
        *,
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        self.url = url
        self._connect = connect or _default_connect
        self._ws: Any | None = None
        self.stable_parts: list[str] = []
        self.partial = ""

    def start(self) -> None:
        self._ws = self._connect(self.url)
        handshake = self._recv(5.0)
        if handshake is None:
            raise RuntimeError("Tencent realtime ASR handshake timed out.")
        self._handle(handshake)

    def send_pcm(self, pcm: bytes) -> str:
        if self._ws is None:
            raise RuntimeError("Tencent realtime ASR session is not started.")
        if pcm:
            self._ws.send(pcm)
        self._drain(0.05)
        return self.current_text()

    def finish(self, timeout_s: float = 2.0) -> str:
        if self._ws is None:
            return self.current_text()
        self._ws.send(json.dumps({"type": "end"}))
        deadline = time.perf_counter() + timeout_s
        saw_quiet = False
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            message = self._recv(max(0.05, min(0.25, remaining)))
            if message is None:
                # Do not block the STT thread for the full timeout once a
                # stable sentence is already in hand.
                if self.current_text() and saw_quiet:
                    break
                saw_quiet = True
                continue
            saw_quiet = False
            self._handle(message)
            if message.get("final") == 1:
                break
        return self.current_text()

    def current_text(self) -> str:
        stable = "".join(self.stable_parts).strip()
        return stable or self.partial.strip()

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            logger.debug("Tencent realtime ASR socket close failed", exc_info=True)
        self._ws = None

    def _recv(self, timeout_s: float) -> dict[str, Any] | None:
        if self._ws is None:
            return None
        try:
            raw = self._ws.recv(timeout=timeout_s)
        except TimeoutError:
            return None
        except Exception:
            logger.debug("Tencent realtime ASR recv failed", exc_info=True)
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Tencent realtime ASR returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tencent realtime ASR event must be a JSON object.")
        return parsed

    def _drain(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            message = self._recv(max(0.0, deadline - time.perf_counter()))
            if message is None:
                return
            self._handle(message)

    def _handle(self, message: dict[str, Any]) -> None:
        code = message.get("code")
        if code not in (None, 0):
            raise RuntimeError(
                "Tencent realtime ASR failed "
                f"(code={code!r}): {message.get('message', 'unknown error')}"
            )
        result = message.get("result") or {}
        text = (result.get("voice_text_str") or "").strip()
        slice_type = result.get("slice_type")
        if slice_type == 2 and text:
            self.stable_parts.append(text)
            self.partial = ""
        elif text:
            self.partial = text
