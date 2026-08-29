"""Tests for the offline latency A/B harness.

The harness is a measuring instrument, so these check that it measures rather
than that it flatters: the baseline must reproduce the sentence-only behaviour,
and the playback model must actually report a gap when one exists.
"""

from __future__ import annotations

import math

import scripts.latency_ab_benchmark as bench

_KW = dict(ttft_s=0.35, tokens_per_s=40.0, tts_ttfb_s=0.205, tts_speed=1.2)


def _reply(text: str, lang: str = "en") -> bench.Reply:
    return bench.Reply("t", lang, text)


# ── tokenizer ────────────────────────────────────────────────────────────────


def test_tokenize_round_trips_the_reply():
    for reply in bench.CORPUS:
        assert "".join(bench.tokenize(reply.text)) == reply.text


def test_tokenize_emits_punctuation_separately():
    # The conservative choice: the comma costs its own token, so the clause
    # boundary cannot arrive earlier than a real provider would deliver it.
    assert bench.tokenize("Sure, ok")[:2] == ["Sure", ","]


def test_tokenize_splits_cjk_per_character():
    assert bench.tokenize("好的，是") == ["好", "的", "，", "是"]


# ── playback model ───────────────────────────────────────────────────────────


def test_speech_duration_scales_with_tts_speed():
    slow = bench.speech_duration_s("hello there friend", 1.0)
    fast = bench.speech_duration_s("hello there friend", 2.0)
    assert slow > fast > 0
    assert abs(slow / fast - 2.0) < 1e-6


def test_cjk_speaks_slower_per_character_than_latin():
    assert bench.speech_duration_s("好的呀", 1.0) > bench.speech_duration_s("abc", 1.0)


def test_playback_model_reports_a_gap_when_tts_is_slow():
    # A long TTS first-byte time cannot be covered by a short opening clause.
    reply = _reply("Sure, I can help with that, let me check the weather for you now.")
    timeline = bench.replay(reply, 8, **{**_KW, "tts_ttfb_s": 1.5})
    assert timeline.gaps_s, "a 1.5s TTS TTFB must starve playback"
    assert timeline.total_gap_s > 0


def test_playback_model_reports_no_gap_for_a_single_chunk():
    timeline = bench.replay(_reply("Done."), 8, **_KW)
    assert len(timeline.chunks) == 1
    assert timeline.gaps_s == []


# ── A/B behaviour ────────────────────────────────────────────────────────────


def test_zero_lookahead_reproduces_sentence_only_flushing():
    reply = _reply("Sure, I can help with that, let me check the weather for you now.")
    baseline = bench.replay(reply, 0, **_KW)
    assert baseline.chunks == ["Sure, I can help with that, let me check the weather for you now."]


def test_lookahead_makes_first_audio_earlier_for_a_clause_reply():
    reply = _reply("Sure, I can help with that, let me check the weather for you now.")
    baseline = bench.replay(reply, 0, **_KW)
    optimized = bench.replay(reply, 8, **_KW)
    assert optimized.first_audio_s < baseline.first_audio_s
    assert optimized.chunks[0] == "Sure,"


def test_reply_without_clause_punctuation_is_unchanged():
    reply = _reply("Let me look that up for you right now.")
    baseline = bench.replay(reply, 0, **_KW)
    optimized = bench.replay(reply, 8, **_KW)
    assert optimized.first_audio_s == baseline.first_audio_s
    assert optimized.chunks == baseline.chunks


def test_thousands_separator_is_not_an_early_flush_point():
    optimized = bench.replay(_reply("一共是1,299元，含税。", lang="zh"), 8, **_KW)
    assert optimized.chunks[0] == "一共是1,299元，"


def test_virtual_clock_is_deterministic():
    reply = _reply("Sure, I can help with that, let me check the weather for you now.")
    first = bench.replay(reply, 8, **_KW)
    second = bench.replay(reply, 8, **_KW)
    assert first.first_audio_s == second.first_audio_s
    assert first.submitted_s == second.submitted_s


def test_slower_token_rate_increases_the_saving():
    reply = _reply("Sure, I can help with that, let me check the weather for you now.")

    def saved(tokens_per_s: float) -> float:
        kwargs = {**_KW, "tokens_per_s": tokens_per_s}
        return bench.replay(reply, 0, **kwargs).first_audio_s - bench.replay(reply, 8, **kwargs).first_audio_s

    assert saved(20.0) > saved(40.0) > 0


# ── hedge model ──────────────────────────────────────────────────────────────


def test_lognormal_fit_round_trips_its_percentiles():
    mu, sigma = bench.lognormal_from_percentiles(878.0, 4664.0)
    assert abs(math.exp(mu) - 878.0) < 1e-6
    assert abs(math.exp(mu + 1.6448536269514722 * sigma) - 4664.0) < 1e-3
