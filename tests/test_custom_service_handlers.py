import base64
import io
import json
import sys
import wave
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from speech_to_speech.arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import FasterWhisperSTTHandlerArguments
from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
from speech_to_speech.arguments_classes.mlx_audio_whisper_arguments import MLXAudioWhisperSTTHandlerArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import ParakeetTDTSTTHandlerArguments
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    EndOfResponse,
    PartialTranscription,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.s2s_pipeline import (
    enable_tencent_realtime_transcription,
    get_stt_handler,
    get_tts_handler,
    parse_arguments,
)
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.STT.tencent_realtime import build_realtime_url
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler


class FakeTencentClient:
    def __init__(self, text="识别成功。"):
        self.text = text
        self.requests = []

    def SentenceRecognition(self, request):
        self.requests.append(request)
        return SimpleNamespace(Result=self.text)


class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body
        self.raise_for_status_calls = 0

    def raise_for_status(self):
        self.raise_for_status_calls += 1

    def json(self):
        return self.body


class FakeStreamResponse:
    def __init__(self, events=None, *, chunks=None, iter_text_fn=None, error=None):
        self.events = events or []
        self.chunks = chunks
        self.iter_text_fn = iter_text_fn
        self.error = error
        self.raise_for_status_calls = 0
        self.closed = False

    def raise_for_status(self):
        self.raise_for_status_calls += 1
        if self.error:
            raise self.error

    def iter_text(self):
        if self.iter_text_fn is not None:
            yield from self.iter_text_fn()
            return
        if self.chunks is not None:
            yield from self.chunks
            return
        for event in self.events:
            if isinstance(event, str):
                yield event
            else:
                yield f"data: {json.dumps(event)}\n\n"

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeHTTPClient:
    def __init__(self, response=None, on_post=None, on_stream=None, stream_response=None):
        self.response = response
        self.stream_response = stream_response
        self.on_post = on_post
        self.on_stream = on_stream
        self.calls = []
        self.closed = False

    def post(self, url, *, headers, json):
        self.calls.append((url, headers, json))
        if self.on_post:
            self.on_post()
        return self.response

    def stream(self, method, url, *, headers, json):
        self.calls.append((url, headers, json))
        if self.on_stream:
            self.on_stream()
        return self.stream_response if self.stream_response is not None else self.response

    def close(self):
        self.closed = True


def _wav_hex(samples, sample_rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue().hex()


def _pcm_hex(samples):
    return np.asarray(samples, dtype="<i2").tobytes().hex()


def _sse_event(samples, status=1, status_code=0, status_msg="success"):
    return {
        "data": {"audio": _pcm_hex(samples), "status": status},
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def _minimax_response(samples):
    return FakeHTTPResponse(
        {
            "data": {"audio": _wav_hex(samples), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
    )


def _tencent_handler(client):
    return TencentASRHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={
            "client": client,
            "engine": "16k_zh",
            "language_code": "zh",
        },
    )


def _minimax_handler(client, cancel_scope=None, stream=True):
    return MiniMaxTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "api_key": "test-key",
            "voice_id": "test-voice",
            "client": client,
            "cancel_scope": cancel_scope,
            "stream": stream,
        },
    )


def test_tencent_asr_converts_float_audio_and_preserves_turn_metadata(monkeypatch):
    client = FakeTencentClient()
    handler = _tencent_handler(client)
    monkeypatch.setattr("speech_to_speech.STT.tencent_asr_handler.console.print", lambda *args, **kwargs: None)
    audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_message = VADAudio(
        audio=audio,
        mode="final",
        turn_id="turn-1",
        turn_revision=2,
    )

    result = list(handler.process(input_message))

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "识别成功。"
    assert result[0].language_code == "zh"
    assert result[0].turn_id == "turn-1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == input_message.created_at_s

    request = client.requests[0]
    assert request["EngSerViceType"] == "16k_zh"
    assert request["SourceType"] == 1
    assert request["VoiceFormat"] == "pcm"
    assert request["DataLen"] == len(audio) * 2
    assert base64.b64decode(request["Data"]) == (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def test_tencent_asr_ignores_short_progressive_audio():
    client = FakeTencentClient()
    handler = _tencent_handler(client)

    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(1600, dtype=np.float32),
                mode="progressive",
            )
        )
    )

    assert result == []
    assert client.requests == []
    handler.cleanup()


def test_tencent_asr_reuses_speculative_result_when_final_tail_is_short(monkeypatch):
    client = FakeTencentClient()
    handler = _tencent_handler(client)
    monkeypatch.setattr("speech_to_speech.STT.tencent_asr_handler.console.print", lambda *args, **kwargs: None)
    progressive = VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=1,
    )
    final = VADAudio(
        audio=np.zeros(16000 + 1600, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=1,
    )

    assert list(handler.process(progressive)) == []
    assert handler._speculative is not None
    handler._speculative["future"].result(timeout=1)

    result = list(handler.process(final))

    assert len(client.requests) == 1
    assert len(result) == 1
    assert result[0].text == "识别成功。"
    handler.cleanup()


def test_tencent_asr_reruns_when_final_audio_is_much_longer(monkeypatch):
    client = FakeTencentClient()
    handler = _tencent_handler(client)
    monkeypatch.setattr("speech_to_speech.STT.tencent_asr_handler.console.print", lambda *args, **kwargs: None)
    progressive = VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=1,
    )
    final = VADAudio(
        audio=np.zeros(16000 * 2, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=1,
    )

    assert list(handler.process(progressive)) == []
    handler._speculative["future"].result(timeout=1)
    result = list(handler.process(final))

    assert len(client.requests) == 2
    assert result[0].text == "识别成功。"
    handler.cleanup()


class FakeRealtimeSession:
    def __init__(self, finish_text="你好世界。"):
        self.pcm_chunks = []
        self.started = False
        self.finished = False
        self.closed = False
        self.finish_text = finish_text
        self.stable_parts = ["你好"]
        self.partial = "你好"

    def start(self):
        self.started = True

    def send_pcm(self, pcm):
        self.pcm_chunks.append(bytes(pcm))
        return self.partial

    def current_text(self):
        return self.partial

    def finish(self, timeout_s=5.0):
        self.finished = True
        return self.finish_text

    def close(self):
        self.closed = True


def _realtime_handler(session):
    return TencentASRHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={
            "client": FakeTencentClient(),
            "engine": "16k_zh",
            "language_code": "zh",
            "realtime_session_factory": lambda: session,
        },
    )


def test_tencent_realtime_url_is_signed_and_uses_pcm():
    url = build_realtime_url(
        app_id="1259228442",
        secret_id="secret-id",
        secret_key="secret-key",
        engine="16k_zh",
        voice_id="voice-1",
        timestamp=1673408372,
        nonce=1673408372,
        expired=1673494772,
    )
    assert url.startswith("wss://asr.cloud.tencent.com/asr/v2/1259228442?")
    assert "voice_format=1" in url
    assert "needvad=0" in url
    assert "engine_model_type=16k_zh" in url
    assert "signature=" in url
    again = build_realtime_url(
        app_id="1259228442",
        secret_id="secret-id",
        secret_key="secret-key",
        engine="16k_zh",
        voice_id="voice-1",
        timestamp=1673408372,
        nonce=1673408372,
        expired=1673494772,
    )
    assert url == again


def test_tencent_realtime_streams_progressive_then_finalizes(monkeypatch):
    session = FakeRealtimeSession()
    handler = _realtime_handler(session)
    monkeypatch.setattr("speech_to_speech.STT.tencent_asr_handler.console.print", lambda *args, **kwargs: None)
    first = np.zeros(3200, dtype=np.float32)
    second = np.zeros(6400, dtype=np.float32)

    partials = list(
        handler.process(
            VADAudio(audio=first, mode="progressive", turn_id="turn-1", turn_revision=1)
        )
    )
    more = list(
        handler.process(
            VADAudio(audio=second, mode="progressive", turn_id="turn-1", turn_revision=1)
        )
    )
    final = list(
        handler.process(
            VADAudio(audio=second, mode="final", turn_id="turn-1", turn_revision=1)
        )
    )

    assert session.started is True
    assert session.finished is True
    assert session.closed is True
    assert all(isinstance(item, PartialTranscription) for item in partials + more)
    assert isinstance(final[0], Transcription)
    assert final[0].text == "你好世界。"
    sent = b"".join(session.pcm_chunks)
    assert len(sent) == 6400 * 2
    handler.cleanup()


def test_tencent_realtime_session_collects_stable_text():
    from speech_to_speech.STT.tencent_realtime import TencentRealtimeASRSession

    class FakeWS:
        def __init__(self):
            self.sent = []
            self.inbox = [
                json.dumps({"code": 0, "message": "ok"}),
                json.dumps({"code": 0, "result": {"slice_type": 1, "voice_text_str": "你"}}),
                json.dumps({"code": 0, "result": {"slice_type": 2, "voice_text_str": "你好。"}}),
                json.dumps({"code": 0, "final": 1}),
            ]

        def recv(self, timeout=None):
            if not self.inbox:
                raise TimeoutError
            return self.inbox.pop(0)

        def send(self, data):
            self.sent.append(data)

        def close(self):
            pass

    socket = FakeWS()
    session = TencentRealtimeASRSession("wss://example.test", connect=lambda _url: socket)
    session.start()
    session.send_pcm(b"\x00\x00")
    text = session.finish()
    session.close()

    assert text == "你好。"
    assert socket.sent[0] == b"\x00\x00"
    assert json.loads(socket.sent[1]) == {"type": "end"}


def test_tencent_realtime_finish_returns_after_quiet_stable_text():
    from speech_to_speech.STT.tencent_realtime import TencentRealtimeASRSession

    class FakeWS:
        def __init__(self):
            self.sent = []
            self.inbox = [
                json.dumps({"code": 0, "message": "ok"}),
                json.dumps({"code": 0, "result": {"slice_type": 2, "voice_text_str": "好的。"}}),
            ]

        def recv(self, timeout=None):
            if self.inbox:
                return self.inbox.pop(0)
            raise TimeoutError

        def send(self, data):
            self.sent.append(data)

        def close(self):
            pass

    session = TencentRealtimeASRSession("wss://example.test", connect=lambda _url: FakeWS())
    session.start()
    session.send_pcm(b"\x00\x00")
    text = session.finish()
    session.close()
    assert text == "好的。"


def test_enable_tencent_realtime_transcription_sets_live_flag(monkeypatch):
    monkeypatch.setenv("TENCENT_ASR_APP_ID", "1")
    args = ModuleArguments(stt="tencent", enable_live_transcription=False)
    enable_tencent_realtime_transcription(args)
    assert args.enable_live_transcription is True
    assert args.live_transcription_update_interval == 0.2


def test_vad_silence_prefetch_emits_once():
    import torch

    from speech_to_speech.VAD.vad_handler import VADHandler

    handler = object.__new__(VADHandler)
    handler.sample_rate = 16000
    handler.min_speech_ms = 384
    handler.min_speech_continuation_ms = 192
    handler.enable_realtime_transcription = False
    handler.speculative_turns = None
    handler._silence_prefetch_emitted = False
    handler._log_progressive_yields = 0
    handler._speculative_audio_prefix = None
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._pending_reopen_candidate = None
    handler._total_samples = 32000
    audio = torch.ones(8000)
    handler.iterator = SimpleNamespace(
        triggered=True,
        temp_end=16000,
        buffer=[audio],
        speech_buffer=lambda: [audio],
        active_speech_samples=8000,
    )

    first = list(handler._maybe_yield_silence_prefetch())
    second = list(handler._maybe_yield_silence_prefetch())

    assert len(first) == 1
    assert first[0].mode == "progressive"
    assert first[0].turn_id == "turn_1"
    assert first[0].turn_revision == 0
    assert second == []


def test_tencent_asr_rejects_audio_over_sentence_limit():
    handler = _tencent_handler(FakeTencentClient())

    with pytest.raises(ValueError, match="at most 60 seconds"):
        list(
            handler.process(
                VADAudio(
                    audio=np.zeros(16000 * 60 + 1, dtype=np.float32),
                    mode="final",
                )
            )
        )


def test_minimax_tts_streams_pcm_and_yields_padded_chunks(monkeypatch):
    samples = np.arange(700, dtype=np.int16)
    stream_response = FakeStreamResponse([_sse_event(samples)])
    client = FakeHTTPClient(stream_response=stream_response)
    handler = _minimax_handler(client)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="你好", language_code="zh")))

    assert len(result) == 2
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in result)
    np.testing.assert_array_equal(result[0], samples[:512])
    np.testing.assert_array_equal(result[1][:188], samples[512:])
    np.testing.assert_array_equal(result[1][188:], np.zeros(324, dtype=np.int16))

    url, headers, payload = client.calls[0]
    assert url == "https://api.minimax.io/v1/t2a_v2"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Accept"] == "text/event-stream"
    assert payload["model"] == "speech-2.8-turbo"
    assert payload["text"] == "你好"
    assert payload["stream"] is True
    assert payload["output_format"] == "hex"
    assert payload["voice_setting"]["voice_id"] == "test-voice"
    assert payload["audio_setting"] == {
        "sample_rate": 16000,
        "format": "pcm",
        "channel": 1,
    }
    assert stream_response.raise_for_status_calls == 1


def test_minimax_tts_yields_short_first_frame_immediately(monkeypatch):
    first = np.arange(120, dtype=np.int16)
    second = np.arange(120, 632, dtype=np.int16)
    released = Event()

    def iter_text():
        yield f"data: {json.dumps(_sse_event(first))}\n\n"
        released.wait(timeout=1)
        yield f"data: {json.dumps(_sse_event(second))}\n\n"

    handler = _minimax_handler(FakeHTTPClient(stream_response=FakeStreamResponse(iter_text_fn=iter_text)))
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    gen = handler.process(TTSInput(text="hello"))
    first_chunk = next(gen)
    np.testing.assert_array_equal(first_chunk, first)
    released.set()
    rest = list(gen)
    assert len(rest) == 1
    np.testing.assert_array_equal(rest[0], second)


def test_minimax_tts_yields_first_chunk_before_stream_ends(monkeypatch):
    first = np.arange(512, dtype=np.int16)
    second = np.arange(512, 1024, dtype=np.int16)
    released = Event()

    def iter_text():
        yield f"data: {json.dumps(_sse_event(first))}\n\n"
        released.wait(timeout=1)
        yield f"data: {json.dumps(_sse_event(second))}\n\n"

    stream_response = FakeStreamResponse(iter_text_fn=iter_text)
    handler = _minimax_handler(FakeHTTPClient(stream_response=stream_response))
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    gen = handler.process(TTSInput(text="hello"))
    first_chunk = next(gen)
    np.testing.assert_array_equal(first_chunk, first)
    released.set()
    rest = list(gen)
    assert len(rest) == 1
    np.testing.assert_array_equal(rest[0], second)


def test_minimax_tts_skips_aggregated_status_two_audio(monkeypatch):
    incremental = np.arange(512, dtype=np.int16)
    aggregated = np.arange(1024, dtype=np.int16)
    stream_response = FakeStreamResponse(
        [
            _sse_event(incremental, status=1),
            _sse_event(aggregated, status=2),
        ]
    )
    handler = _minimax_handler(FakeHTTPClient(stream_response=stream_response))
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], incremental)


def test_minimax_tts_reassembles_split_hex_frames(monkeypatch):
    samples = np.arange(512, dtype=np.int16)
    hex_audio = _pcm_hex(samples)
    midpoint = (len(hex_audio) // 2) | 1  # odd split so a nibble is carried
    first_hex = hex_audio[:midpoint]
    second_hex = hex_audio[midpoint:]
    events = [
        {
            "data": {"audio": first_hex, "status": 1},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
        {
            "data": {"audio": second_hex, "status": 1},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    ]
    handler = _minimax_handler(FakeHTTPClient(stream_response=FakeStreamResponse(events)))
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    played = np.concatenate(result)
    np.testing.assert_array_equal(played[: len(samples)], samples)


def test_minimax_tts_emits_end_of_response_sentinel():
    handler = _minimax_handler(FakeHTTPClient(stream_response=FakeStreamResponse([])))

    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_minimax_tts_surfaces_provider_errors():
    stream_response = FakeStreamResponse(
        [
            {
                "data": None,
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
            }
        ]
    )
    handler = _minimax_handler(FakeHTTPClient(stream_response=stream_response))

    with pytest.raises(RuntimeError, match="invalid api key"):
        list(handler.process(TTSInput(text="hello")))


def test_minimax_tts_drops_audio_after_interruption(monkeypatch):
    cancel_scope = CancelScope()
    stream_response = FakeStreamResponse([_sse_event(np.arange(600, dtype=np.int16))])
    client = FakeHTTPClient(stream_response=stream_response, on_stream=cancel_scope.cancel)
    handler = _minimax_handler(client, cancel_scope=cancel_scope)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert list(handler.process(TTSInput(text="hello"))) == []


def test_minimax_tts_cancels_after_first_streamed_chunk(monkeypatch):
    cancel_scope = CancelScope()
    first = np.arange(512, dtype=np.int16)
    second = np.arange(512, 1024, dtype=np.int16)

    def iter_text():
        yield f"data: {json.dumps(_sse_event(first))}\n\n"
        cancel_scope.cancel()
        yield f"data: {json.dumps(_sse_event(second))}\n\n"

    handler = _minimax_handler(
        FakeHTTPClient(stream_response=FakeStreamResponse(iter_text_fn=iter_text)),
        cancel_scope=cancel_scope,
    )
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], first)


def test_minimax_tts_nonstream_validates_returned_sample_rate(monkeypatch):
    response = FakeHTTPResponse(
        {
            "data": {"audio": _wav_hex(np.zeros(10, dtype=np.int16), sample_rate=24000)},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
    )
    handler = _minimax_handler(FakeHTTPClient(response), stream=False)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="24000 Hz"):
        list(handler.process(TTSInput(text="hello")))


def test_minimax_tts_nonstream_sends_wav_payload(monkeypatch):
    samples = np.arange(512, dtype=np.int16)
    response = _minimax_response(samples)
    client = FakeHTTPClient(response)
    handler = _minimax_handler(client, stream=False)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], samples)
    _url, _headers, payload = client.calls[0]
    assert payload["stream"] is False
    assert payload["audio_setting"]["format"] == "wav"


def test_minimax_cleanup_does_not_close_injected_client():
    client = FakeHTTPClient(stream_response=FakeStreamResponse([]))
    handler = _minimax_handler(client)

    handler.cleanup()

    assert client.closed is False


def test_custom_service_json_profile_selects_all_three_providers():
    original_argv = sys.argv[:]
    profile = Path(__file__).parents[1] / "configs" / "tencent-deepseek-minimax.json"
    try:
        sys.argv = ["speech-to-speech", str(profile)]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert args.module_kwargs.mode == "realtime"
    assert args.module_kwargs.stt == "tencent"
    assert args.module_kwargs.enable_live_transcription is False
    assert args.module_kwargs.llm_backend == "chat-completions"
    assert args.module_kwargs.tts == "minimax"
    assert args.responses_api_language_model_handler_kwargs.model_name == "deepseek-v4-flash"
    assert args.responses_api_language_model_handler_kwargs.responses_api_base_url == "https://api.deepseek.com"
    assert args.responses_api_language_model_handler_kwargs.responses_api_disable_thinking is False
    assert args.responses_api_language_model_handler_kwargs.stream_batch_sentences == 1
    assert args.vad_handler_kwargs.speech_pad_ms == 80


def test_get_stt_handler_builds_tencent_adapter(monkeypatch):
    monkeypatch.setattr(TencentASRHandler, "setup", lambda self: None)

    handler = get_stt_handler(
        ModuleArguments(stt="tencent"),
        Event(),
        Queue(),
        Queue(),
        None,
        WhisperSTTHandlerArguments(),
        FasterWhisperSTTHandlerArguments(),
        ParaformerSTTHandlerArguments(),
        MLXAudioWhisperSTTHandlerArguments(),
        ParakeetTDTSTTHandlerArguments(),
    )

    assert isinstance(handler, TencentASRHandler)


def test_get_tts_handler_builds_minimax_adapter_with_runtime_guards(monkeypatch):
    recorded = {}

    def fake_setup(self, should_listen, cancel_scope=None, speculative_turns=None):
        recorded["should_listen"] = should_listen
        recorded["cancel_scope"] = cancel_scope
        recorded["speculative_turns"] = speculative_turns

    monkeypatch.setattr(MiniMaxTTSHandler, "setup", fake_setup)
    should_listen = Event()
    cancel_scope = CancelScope()
    speculative_turns = SimpleNamespace()

    handler = get_tts_handler(
        ModuleArguments(tts="minimax"),
        Event(),
        Queue(),
        Queue(),
        should_listen,
        ChatTTSHandlerArguments(),
        FacebookMMSTTSHandlerArguments(),
        PocketTTSHandlerArguments(),
        KokoroTTSHandlerArguments(),
        Qwen3TTSHandlerArguments(),
        cancel_scope=cancel_scope,
        speculative_turns=speculative_turns,
    )

    assert isinstance(handler, MiniMaxTTSHandler)
    assert recorded == {
        "should_listen": should_listen,
        "cancel_scope": cancel_scope,
        "speculative_turns": speculative_turns,
    }
