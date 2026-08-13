from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    AudioOutput,
    GenerateResponseRequest,
    LLMResponseChunk,
    PartialTranscription,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.STT.tencent_realtime_client import TencentRecognitionResult
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler


class _FakeCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=None,
        )


class _FakeClient:
    def __init__(self, *_args, **_kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    def with_options(self, **_kwargs):
        return self


class _ControlledStream:
    def __init__(self, terminal_gate: threading.Event, first_text: str = "First sentence. Second"):
        self.terminal_gate = terminal_gate
        self.first_text = first_text
        self.closed = threading.Event()

    def __iter__(self):
        yield _chat_chunk(self.first_text)
        self.terminal_gate.wait(timeout=1.0)
        yield _chat_chunk(" sentence.")

    def close(self):
        self.closed.set()


class _CloseReleasedStream:
    def __init__(self, *, raise_after_close: bool = False, close_delay_s: float = 0.0):
        self.first_delta = threading.Event()
        self.blocked = threading.Event()
        self.closed = threading.Event()
        self.raise_after_close = raise_after_close
        self.close_delay_s = close_delay_s
        self.close_calls = 0
        self.concurrent_closes = 0
        self.max_concurrent_closes = 0
        self._close_lock = threading.Lock()

    def __iter__(self):
        self.first_delta.set()
        yield _chat_chunk("Leading fragment")
        self.blocked.set()
        self.closed.wait(timeout=1.0)
        if self.raise_after_close:
            raise RuntimeError("stream closed by cancellation")
        yield _chat_chunk(" STALE_PROVIDER_TEXT")

    def close(self):
        with self._close_lock:
            self.close_calls += 1
            self.concurrent_closes += 1
            self.max_concurrent_closes = max(self.max_concurrent_closes, self.concurrent_closes)
        try:
            self.closed.set()
            if self.close_delay_s:
                time.sleep(self.close_delay_s)
        finally:
            with self._close_lock:
                self.concurrent_closes -= 1


def _chat_chunk(text: str):
    delta = SimpleNamespace(content=text, refusal=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _handler(**setup_overrides):
    original = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        setup = {
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "sentinel-api-key",
            "stream": True,
            "disable_thinking": True,
            "compact_history": False,
        }
        setup.update(setup_overrides)
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=setup,
        )
    finally:
        base_mod.OpenAI = original


def _request() -> GenerateResponseRequest:
    chat = Chat(10)
    chat.add_item(make_user_message("hello"))
    runtime = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Be concise."),
    )
    return GenerateResponseRequest(runtime_config=runtime, turn_id="turn-1", turn_revision=2)


class _Clock:
    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class _ManualClock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_first_sentence_reaches_real_tts_boundary_before_provider_terminal(monkeypatch):
    profile = Path(__file__).parents[1] / "configs" / "tencent-deepseek-minimax.json"
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", str(profile)])
    args = parse_arguments()

    assert args.responses_api_language_model_handler_kwargs.stream_batch_sentences == 1
    assert args.responses_api_language_model_handler_kwargs.responses_api_stream is True
    assert args.responses_api_language_model_handler_kwargs.responses_api_disable_thinking is True

    terminal_gate = threading.Event()
    stream = _ControlledStream(terminal_gate)
    handler = _handler(stream_batch_sentences=1)
    handler.client.chat.completions.create = lambda **_kwargs: stream
    processor = LMOutputProcessor(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={"text_output_queue": queue.Queue()},
    )

    generation = handler.process(_request())
    try:
        first = next(generation)
        assert isinstance(first, LLMResponseChunk)
        assert first.text == "First sentence."
        assert first.turn_id == "turn-1"
        assert first.turn_revision == 2
        tts_inputs = list(processor.process(first))
        assert len(tts_inputs) == 1
        assert tts_inputs[0].text == "First sentence."
        assert tts_inputs[0].speakable_phrase_at_s == first.speakable_phrase_at_s
        assert terminal_gate.is_set() is False
    finally:
        terminal_gate.set()
        list(generation)


def test_cancel_closes_blocked_deepseek_stream_without_stale_output(caplog):
    cancel_scope = CancelScope(clock=lambda: 3.0)
    stream = _CloseReleasedStream(raise_after_close=True, close_delay_s=0.05)
    handler = _handler(stream_batch_sentences=1, cancel_scope=cancel_scope, clock=_Clock(1.0, 2.0, 3.05))
    handler.client.chat.completions.create = lambda **_kwargs: stream
    outputs: list[object] = []
    completed = threading.Event()

    def consume():
        outputs.extend(handler.process(_request()))
        completed.set()

    with caplog.at_level(logging.INFO):
        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        assert stream.first_delta.wait(timeout=0.2)
        assert stream.blocked.wait(timeout=0.2)
        try:
            cancel_scope.cancel()
            assert stream.closed.wait(timeout=0.2)
            assert completed.wait(timeout=0.2)
        finally:
            if not stream.closed.is_set():
                stream.close()
            worker.join(timeout=1.0)

    assert all("STALE_PROVIDER_TEXT" not in output.text for output in outputs if isinstance(output, LLMResponseChunk))
    ends = [output for output in outputs if output.tag == "end_of_response"]
    assert len(ends) == 1 and ends[0].error is None
    assert stream.close_calls == 1
    assert stream.max_concurrent_closes == 1
    assert "Provider stream barge-in close latency: 0.050s (turn=turn-1 rev=2)" in caplog.text


def test_latency_metrics_and_timestamps_are_monotonic_and_content_free(caplog):
    transcript = "SENTINEL_TRANSCRIPT"
    api_key = "SENTINEL_API_KEY"
    signature = "SENTINEL_SIGNATURE"
    audio_hex = "53454e54494e454c5f415544494f"
    raw_json = '{"provider":"SENTINEL_RAW_JSON"}'

    class Session:
        def __init__(self):
            self.results = [TencentRecognitionResult(transcript, final=False, stable=False)]

        def push_snapshot(self, _audio):
            pass

        def finish(self, _audio):
            self.results = [TencentRecognitionResult(transcript, final=True, stable=True)]

        def drain_results(self):
            results, self.results = self.results, []
            return results

        def close(self):
            pass

    session = Session()
    stt = TencentASRHandler(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={
            "app_id": "app",
            "secret_id": "id",
            "secret_key": signature,
            "session_factory": lambda _config: session,
            "clock": _Clock(12.0, 23.0),
        },
    )
    llm = _handler(stream_batch_sentences=1, clock=_Clock(30.0, 31.5, 32.0))
    llm.client.chat.completions.create = lambda **_kwargs: _ControlledStream(
        threading.Event(),
        first_text=raw_json + ". Second",
    )

    class TTSClient:
        def start(self, *, cancelled):
            assert cancelled() is False

        def synthesize(self, _text, *, cancelled):
            assert cancelled() is False
            yield np.frombuffer(bytes.fromhex(audio_hex), dtype=np.uint8)

        def close(self, *, graceful=False):
            pass

    tts = MiniMaxTTSHandler(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_args=(threading.Event(),),
        setup_kwargs={
            "api_key": api_key,
            "voice_id": "voice",
            "client_factory": lambda _config: TTSClient(),
            "clock": _Clock(32.1, 32.4, 33.0),
        },
    )

    caplog.clear()
    notifier = TranscriptionNotifier(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={"text_output_queue": queue.Queue()},
    )
    processor = LMOutputProcessor(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs={"text_output_queue": queue.Queue()},
    )
    with caplog.at_level(logging.DEBUG):
        partial = list(
            stt.process(
                VADAudio(
                    audio=np.zeros(1),
                    mode="progressive",
                    turn_id="turn-1",
                    turn_revision=2,
                    created_at_s=10.0,
                )
            )
        )[0]
        final = list(
            stt.process(
                VADAudio(
                    audio=np.zeros(1),
                    mode="final",
                    turn_id="turn-1",
                    turn_revision=2,
                    created_at_s=20.0,
                )
            )
        )[0]
        list(notifier.process(partial))
        list(notifier.process(final))
        llm_chunk = next(llm.process(_request()))
        list(processor.process(llm_chunk))
        tts_input = TTSInput(
            text=transcript,
            turn_id="turn-1",
            turn_revision=2,
            speech_stopped_at_s=31.0,
            cancel_generation=7,
            speakable_phrase_at_s=llm_chunk.speakable_phrase_at_s,
        )
        first_audio = next(tts.process(tts_input))
        queued_audio = tts.output_for_queue(first_audio, tts_input)

    assert isinstance(partial, PartialTranscription) and partial.first_partial_at_s == 12.0
    assert isinstance(final, Transcription) and final.final_at_s == 23.0
    assert isinstance(llm_chunk, LLMResponseChunk) and llm_chunk.first_delta_at_s == 31.5
    assert isinstance(queued_audio, AudioOutput) and queued_audio.first_audio_at_s == 33.0
    assert "Tencent first partial latency: 2.000s (turn=turn-1 rev=2)" in caplog.text
    assert "Tencent final latency: 3.000s (turn=turn-1 rev=2)" in caplog.text
    assert "Provider stream first delta latency: 1.500s (turn=turn-1 rev=2)" in caplog.text
    assert "MiniMax phrase-ready to dispatch latency: 0.400s (turn=turn-1 rev=2)" in caplog.text
    assert "MiniMax request to first audio latency: 0.900s (turn=turn-1 rev=2)" in caplog.text
    assert "Speech end to first audio latency: 2.000s (turn=turn-1 rev=2)" in caplog.text
    forbidden = (transcript, api_key, signature, audio_hex, raw_json, "SENTINEL_RAW_JSON")
    assert all(value not in caplog.text for value in forbidden)


def test_timing_fields_keep_internal_message_construction_backward_compatible():
    assert PartialTranscription(text="partial").first_partial_at_s is None
    assert Transcription(text="final").final_at_s is None
    assert LLMResponseChunk(text="delta").first_delta_at_s is None
    assert LLMResponseChunk(text="delta").speakable_phrase_at_s is None
    assert TTSInput(text="phrase").speakable_phrase_at_s is None
    assert AudioOutput(audio=b"pcm").first_audio_at_s is None


def test_minimax_barge_in_records_closure_and_last_accepted_audio(caplog):
    cancel_clock = _ManualClock(10.0)
    cancel_scope = CancelScope(clock=cancel_clock)

    class Client:
        def start(self, *, cancelled):
            assert cancelled() is False

        def synthesize(self, _text, *, cancelled):
            yield np.ones(512, dtype=np.float32)
            if cancelled():
                return
            yield np.ones(512, dtype=np.float32)

        def close(self, *, graceful=False):
            pass

    handler = MiniMaxTTSHandler(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_args=(threading.Event(),),
        setup_kwargs={
            "api_key": "key",
            "voice_id": "voice",
            "cancel_scope": cancel_scope,
            "client_factory": lambda _config: Client(),
            "clock": _Clock(10.0, 10.1, 10.2, 12.3),
        },
    )
    generation = handler.process(
        TTSInput(
            text="phrase",
            turn_id="turn-1",
            turn_revision=2,
            speakable_phrase_at_s=9.9,
        )
    )

    with caplog.at_level(logging.INFO):
        next(generation)
        cancel_clock.now = 12.0
        cancel_scope.cancel()
        assert list(generation) == []

    assert "MiniMax barge-in close latency: 0.300s (turn=turn-1 rev=2)" in caplog.text
    assert "MiniMax last accepted audio offset from barge-in: -1.800s (turn=turn-1 rev=2)" in caplog.text
