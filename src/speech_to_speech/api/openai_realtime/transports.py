"""Session transports: how server events and outbound audio reach a client.

The per-unit send loop in ``websocket_router`` is transport-agnostic: it owns
the pipeline output queues (sentinels, generation discards, SESSION_END drain)
and hands client-visible traffic to the transport attached to the current
``SessionState``. Two implementations exist:

- ``WebSocketTransport`` (here): events and base64 audio deltas as JSON frames.
- ``WebRTCSession`` (in ``webrtc_session``, requires the ``webrtc`` extra):
  events over the ``oai-events`` data channel, audio over the RTP media track.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent


class TransportError(ConnectionError):
    """The peer cannot accept an event that the service attempted to deliver."""


class SessionTransport(ABC):
    """What the send loop and client-event dispatch need from a transport."""

    kind: str

    @abstractmethod
    async def send_events(self, events: list[ServerEvent]) -> None: ...

    @abstractmethod
    async def send_audio_chunk(self, service: RealtimeService, session_id: str, pcm: bytes) -> None:
        """Deliver a pipeline-rate PCM16 chunk to the client."""

    @abstractmethod
    def discard_pending_audio(self) -> None:
        """Drop transport-buffered audio that has not reached the client yet.

        WebSocket clients buffer audio on their side, so this is a no-op there;
        the WebRTC transport paces playback server-side and must flush its
        track buffer for barge-in to actually silence the assistant.
        """

    @abstractmethod
    async def close(self) -> None: ...


async def send_ws_event(ws: WebSocket, event: ServerEvent) -> None:
    if ws.application_state != WebSocketState.CONNECTED:
        raise TransportError("WebSocket is not connected")
    try:
        await ws.send_json(event.model_dump())
    except WebSocketDisconnect as exc:
        raise TransportError("WebSocket disconnected during send") from exc
    except RuntimeError as e:
        raise TransportError("WebSocket rejected an event send") from e
    except Exception as exc:  # noqa: BLE001
        raise TransportError(f"WebSocket event send failed ({type(exc).__name__})") from exc


class WebSocketTransport(SessionTransport):
    """JSON-over-WebSocket transport: audio is sent as base64 delta events."""

    kind = "websocket"

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def send_events(self, events: list[ServerEvent]) -> None:
        for event in events:
            await send_ws_event(self.websocket, event)

    async def send_audio_chunk(self, service: RealtimeService, session_id: str, pcm: bytes) -> None:
        await self.send_events(service.encode_audio_chunk(session_id, pcm))

    def discard_pending_audio(self) -> None:
        """Discard transport-owned pending output; WebSocket has none.

        The WebSocket client owns playback buffering and must clear it when the
        server emits speech_started/cancellation events.
        """

    async def close(self) -> None:
        try:
            await self.websocket.close()
        except Exception:  # noqa: BLE001
            pass
