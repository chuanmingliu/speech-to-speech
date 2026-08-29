"""Tests for the real-pipeline latency benchmark.

These run the actual handler chain over a real local HTTP provider, so they are
deliberately configured to be fast (tiny TTFT, high token rate, near-zero TTS
first-byte) rather than realistic. The point is that the wiring works and that
the A/B distinction survives the thread and queue hops.
"""

from __future__ import annotations

import pytest
import scripts.latency_pipeline_benchmark as pipe

FAST = dict(ttfb_s=0.005, tts_speed=1.2, use_speculative=True)
_REPLY = pipe.CORPUS[6]  # en-weather: "Sure, I can help with that, ..."


@pytest.fixture(scope="module")
def provider():
    server, port = pipe.start_provider(ttft_s=0.01, tokens_per_s=500.0)
    yield port
    server.shutdown()


def test_pipeline_delivers_the_whole_reply(provider):
    result = pipe.run_once(_REPLY, 0, provider, **FAST)
    assert result.chunks == ["Sure, I can help with that, let me check the weather for you now."]
    assert result.first_audio_s > 0


def test_lookahead_splits_the_opening_clause_through_the_real_chain(provider):
    result = pipe.run_once(_REPLY, 8, provider, **FAST)
    assert result.chunks[0] == "Sure,"
    assert len(result.chunks) == 2


def test_speculative_gate_does_not_swallow_the_first_chunk(provider):
    """The gate can block; make sure an early flush still reaches TTS."""
    gated = pipe.run_once(_REPLY, 8, provider, **FAST)
    ungated = pipe.run_once(_REPLY, 8, provider, **{**FAST, "use_speculative": False})
    assert gated.chunks == ungated.chunks


def test_turn_completes_rather_than_hanging(provider):
    # run_once raises TimeoutError if EndOfResponse never reaches the TTS stage.
    result = pipe.run_once(pipe.CORPUS[3], 8, provider, **FAST)  # zh-terse
    assert result.chunks == ["好的。"]
