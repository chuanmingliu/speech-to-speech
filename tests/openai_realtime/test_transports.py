from __future__ import annotations

import asyncio

import pytest

from speech_to_speech.api.openai_realtime.transports import WebSocketTransport


class _Service:
    def encode_audio_chunk(self, _session_id: str, pcm: bytes) -> list[object]:
        return [pcm]


@pytest.mark.asyncio
async def test_websocket_audio_delivery_never_waits_on_playback_pacing(monkeypatch) -> None:
    transport = WebSocketTransport(object())
    sent: list[list[object]] = []

    async def record(events: list[object]) -> None:
        sent.append(events)

    monkeypatch.setattr(transport, "send_events", record)
    pcm = bytes(6400)

    async def send_twice() -> None:
        await transport.send_audio_chunk(_Service(), "session", pcm)
        await transport.send_audio_chunk(_Service(), "session", pcm)

    await asyncio.wait_for(send_twice(), timeout=0.05)

    assert sent == [[pcm], [pcm]]


def test_discard_pending_websocket_audio_is_safe_and_synchronous() -> None:
    transport = WebSocketTransport(object())

    assert transport.discard_pending_audio() is None
