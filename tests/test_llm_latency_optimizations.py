"""Latency-path unit tests for the OpenAI-compatible LLM handler.

Two optimisations are covered, both driven entirely by in-process fakes:

* the clause-level first flush, which hands TTS an opening clause instead of
  making it wait for the sentence to terminate, and
* request hedging, which races a second completion when the first has produced
  no token in time.

No provider is contacted.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.utils import split_first_spoken_unit
from speech_to_speech.pipeline.messages import GenerateResponseRequest, LLMResponseChunk

# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


ccm.Stream = _FakeStream


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    def with_options(self, **kwargs):
        return self


def _delta(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None), finish_reason=None)],
        usage=None,
    )


def _make_handler(**setup):
    original = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=True,
                init_chat_prompt="Keep answers short.",
                disable_thinking=True,
                compact_history=False,
                **setup,
            ),
        )
    finally:
        base_mod.OpenAI = original


def _spoken_chunks(handler):
    chat = Chat(10)
    chat.add_item(make_user_message("Hallo"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter."),
    )
    request = GenerateResponseRequest(
        runtime_config=runtime_config,
        response=None,
        language_code="de",
        turn_id="t",
        turn_revision=0,
    )
    return [out.text for out in handler.process(request) if isinstance(out, LLMResponseChunk) and out.text]


# ── split_first_spoken_unit ──────────────────────────────────────────────────


def test_first_unit_breaks_on_the_opening_clause():
    assert split_first_spoken_unit("Sure, I can help with that.", 8) == ("Sure,", " I can help with that.")


def test_first_unit_waits_until_enough_text_is_buffered():
    # The clause is there, but nothing follows it yet, so flushing could leave a
    # gap in playback while the model writes the rest.
    assert split_first_spoken_unit("Sure,", 8) == ("", "Sure,")


def test_first_unit_waits_when_no_boundary_has_arrived():
    assert split_first_spoken_unit("Sure I can help with that", 8) == ("", "Sure I can help with that")


def test_first_unit_breaks_on_cjk_clause_punctuation():
    head, rest = split_first_spoken_unit("好的，我帮你查一下今天的天气。", 8)
    assert head == "好的，"
    assert rest == "我帮你查一下今天的天气。"


def test_first_unit_keeps_numbers_intact():
    # A comma inside a thousands separator is not a pause.
    assert split_first_spoken_unit("It costs 1,000 dollars today", 8) == ("", "It costs 1,000 dollars today")
    # Nor is a decimal point that may still be mid-number at the buffer tail.
    assert split_first_spoken_unit("The total is 3.", 8) == ("", "The total is 3.")


def test_first_unit_keeps_abbreviations_and_versioned_names_intact():
    assert split_first_spoken_unit("Use gpt-5.4-mini for now", 8) == ("", "Use gpt-5.4-mini for now")


def test_first_unit_never_flushes_bare_punctuation():
    head, rest = split_first_spoken_unit("，你好，我是助手。", 4)
    assert head == "，你好，"
    assert rest == "我是助手。"


def test_first_unit_disabled_by_zero_lookahead():
    assert split_first_spoken_unit("Sure, I can help with that.", 0) == ("", "Sure, I can help with that.")


# ── streaming first flush ────────────────────────────────────────────────────

# Deltas carry their own leading space, the way real providers stream them.
_REPLY_DELTAS = ["Sure,", " I can", " help you", " with that today."]


def test_first_chunk_flushes_on_a_clause_before_the_sentence_ends():
    handler = _make_handler(stream_first_chunk_lookahead_chars=8, stream_batch_sentences=1)
    handler.client.chat.completions.create = lambda **k: _FakeStream(_delta(d) for d in _REPLY_DELTAS)
    chunks = _spoken_chunks(handler)
    # TTS receives the opening clause instead of waiting for the terminator.
    assert chunks == ["Sure,", "I can help you with that today."]


def test_first_chunk_falls_back_to_sentences_when_disabled():
    handler = _make_handler(stream_first_chunk_lookahead_chars=0, stream_batch_sentences=1)
    handler.client.chat.completions.create = lambda **k: _FakeStream(_delta(d) for d in _REPLY_DELTAS)
    assert _spoken_chunks(handler) == ["Sure, I can help you with that today."]


def test_only_the_first_chunk_uses_a_clause_boundary():
    handler = _make_handler(stream_first_chunk_lookahead_chars=8, stream_batch_sentences=1)
    handler.client.chat.completions.create = lambda **k: _FakeStream(
        _delta(d) for d in ["Sure,", " I can help.", " It rains today,", " all afternoon."]
    )
    chunks = _spoken_chunks(handler)
    assert chunks[0] == "Sure,"
    # Later sentences are flushed whole; their commas are not flush points.
    assert chunks[1:] == ["I can help.", "It rains today, all afternoon."]


def test_first_chunk_lookahead_default_is_enabled():
    assert _make_handler().stream_first_chunk_lookahead_chars > 0


# ── request hedging ──────────────────────────────────────────────────────────


def test_hedge_wins_when_the_first_attempt_stalls():
    handler = _make_handler(request_hedge_after_ms=40)
    attempts: list[int] = []
    started = threading.Event()

    def create(**kwargs):
        index = len(attempts)
        attempts.append(index)
        if index == 0:
            started.set()
            time.sleep(2.0)  # the stalled provider the hedge exists for
            return _FakeStream([_delta("slow answer.")])
        return _FakeStream([_delta("fast answer.")])

    handler.client.chat.completions.create = create
    chunks = _spoken_chunks(handler)
    assert started.is_set()
    assert len(attempts) == 2, "a second attempt should have been issued"
    assert "fast answer." in " ".join(chunks)


def test_no_hedge_is_issued_when_the_first_attempt_answers_in_time():
    handler = _make_handler(request_hedge_after_ms=500)
    attempts: list[int] = []

    def create(**kwargs):
        attempts.append(len(attempts))
        return _FakeStream([_delta("prompt answer.")])

    handler.client.chat.completions.create = create
    chunks = _spoken_chunks(handler)
    assert attempts == [0]
    assert "prompt answer." in " ".join(chunks)


def test_a_failed_first_attempt_is_retried_without_waiting_for_the_timer():
    handler = _make_handler(request_hedge_after_ms=10_000)
    attempts: list[int] = []

    def create(**kwargs):
        index = len(attempts)
        attempts.append(index)
        if index == 0:
            raise RuntimeError("provider blew up")
        return _FakeStream([_delta("second time lucky.")])

    handler.client.chat.completions.create = create
    started_at_s = time.monotonic()
    chunks = _spoken_chunks(handler)
    assert time.monotonic() - started_at_s < 5.0, "the retry must not wait out the hedge timer"
    assert attempts == [0, 1]
    assert "second time lucky." in " ".join(chunks)


def test_hedging_is_off_by_default():
    handler = _make_handler()
    assert handler.request_hedge_after_s == 0.0
