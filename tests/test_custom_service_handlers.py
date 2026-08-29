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
from speech_to_speech.arguments_classes.minimax_tts_arguments import MiniMaxTTSHandlerArguments
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


class FakeHTTPClient:
    def __init__(self, response=None):
        self.response = response
        self.calls = []
        self.closed = False

    def post(self, url, *, headers, json, **_kwargs):
        self.calls.append((url, headers, json))
        return self.response

    def close(self):
        self.closed = True


class FakeWebSocket:
    def __init__(self, messages=None, *, on_continue=None):
        self.messages = list(messages or [])
        self.on_continue = on_continue
        self.sent = []
        self.connect_calls = []
        self.closed = False

    def connect(self, endpoint, *, headers, open_timeout_s):
        self.connect_calls.append((endpoint, headers, open_timeout_s))
        return self

    def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("event") == "task_continue" and self.on_continue:
            self.on_continue()

    def recv(self, timeout=None):
        if not self.messages:
            raise TimeoutError
        message = self.messages.pop(0)
        if callable(message):
            message = message()
        if isinstance(message, BaseException):
            raise message
        return message if isinstance(message, (str, bytes)) else json.dumps(message)

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


def _ws_event(samples=None, *, audio_hex=None, is_final=True, status_code=0, status_msg="success"):
    if audio_hex is None and samples is not None:
        audio_hex = _pcm_hex(samples)
    return {
        "event": "task_continued",
        "data": {"audio": audio_hex} if audio_hex is not None else None,
        "is_final": is_final,
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def _ws_messages(*continued_events):
    return [
        {
            "event": "connected_success",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
        {
            "event": "task_started",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
        *continued_events,
    ]


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


def _minimax_handler(
    client,
    cancel_scope=None,
    stream=True,
    *,
    websocket=None,
    speed=None,
    warmup_connection=None,
    warmup_model=None,
    model_warmup_text=None,
    cache_max_mb=None,
):
    websocket = websocket or FakeWebSocket()
    return MiniMaxTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "api_key": "test-key",
            "model": "speech-2.8-turbo",
            "voice_id": "test-voice",
            "speed": speed,
            "endpoint": "https://api.minimax.io/v1/t2a_v2",
            "websocket_endpoint": "wss://api.minimax.io/ws/v1/t2a_v2",
            "language_boost": "auto",
            "client": client,
            "websocket_connect": websocket.connect,
            "cancel_scope": cancel_scope,
            "stream": stream,
            "warmup_connection": warmup_connection,
            "warmup_model": warmup_model,
            "model_warmup_text": model_warmup_text,
            "cache_max_mb": cache_max_mb,
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
        self.drain_timeouts = []
        self.started = False
        self.finished = False
        self.closed = False
        self.finish_text = finish_text
        self.stable_parts = ["你好"]
        self.partial = "你好"

    def start(self):
        self.started = True

    def send_pcm(self, pcm, *, drain_timeout_s=0.05):
        self.pcm_chunks.append(bytes(pcm))
        self.drain_timeouts.append(drain_timeout_s)
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
    final_audio = np.zeros(7200, dtype=np.float32)

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
            VADAudio(audio=final_audio, mode="final", turn_id="turn-1", turn_revision=1)
        )
    )

    assert session.started is True
    assert session.finished is True
    assert session.closed is True
    assert all(isinstance(item, PartialTranscription) for item in partials + more)
    assert isinstance(final[0], Transcription)
    assert final[0].text == "你好世界。"
    sent = b"".join(session.pcm_chunks)
    assert len(sent) == len(final_audio) * 2
    assert session.drain_timeouts[-1] == 0.0
    handler.cleanup()


def test_tencent_realtime_failure_falls_back_without_retrying_same_turn(monkeypatch):
    class FailingRealtimeSession:
        def __init__(self):
            self.start_calls = 0

        def start(self):
            self.start_calls += 1
            raise TimeoutError("connect timed out")

    session = FailingRealtimeSession()
    handler = _realtime_handler(session)
    monkeypatch.setattr("speech_to_speech.STT.tencent_asr_handler.console.print", lambda *args, **kwargs: None)
    progressive = VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )
    final = VADAudio(
        audio=np.zeros(16000, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )

    assert list(handler.process(progressive)) == []
    assert handler._speculative is not None
    handler._speculative["future"].result(timeout=1)
    result = list(handler.process(final))

    assert session.start_calls == 1
    assert result[0].text == "识别成功。"
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


def test_enable_tencent_realtime_transcription_streams_without_app_id(monkeypatch):
    monkeypatch.delenv("TENCENT_ASR_APP_ID", raising=False)
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
    websocket = FakeWebSocket(_ws_messages(_ws_event(samples)))
    client = FakeHTTPClient()
    handler = _minimax_handler(client, websocket=websocket, speed=1.25)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="你好", language_code="zh")))

    assert len(result) == 2
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in result)
    np.testing.assert_array_equal(result[0], samples[:512])
    np.testing.assert_array_equal(result[1][:188], samples[512:])
    np.testing.assert_array_equal(result[1][188:], np.zeros(324, dtype=np.int16))

    assert client.calls == []
    url, headers, open_timeout_s = websocket.connect_calls[0]
    assert url == "wss://api.minimax.io/ws/v1/t2a_v2"
    assert headers["Authorization"] == "Bearer test-key"
    assert open_timeout_s == 5.0
    task_start, task_continue = websocket.sent
    assert task_start["event"] == "task_start"
    assert task_start["model"] == "speech-2.8-turbo"
    assert task_start["voice_setting"]["voice_id"] == "test-voice"
    assert task_start["voice_setting"]["speed"] == 1.25
    assert task_start["audio_setting"] == {
        "sample_rate": 16000,
        "format": "pcm",
        "channel": 1,
    }
    assert task_start["continuous_sound"] is False
    assert task_continue == {"event": "task_continue", "text": "你好"}


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_minimax_tts_rejects_speed_outside_provider_range(speed):
    with pytest.raises(ValueError, match="between 0.5 and 2.0"):
        _minimax_handler(FakeHTTPClient(), speed=speed)


def test_minimax_tts_warms_persistent_websocket_task():
    websocket = FakeWebSocket(
        _ws_messages()
        + _ws_messages()
    )
    handler = _minimax_handler(
        FakeHTTPClient(),
        websocket=websocket,
        warmup_connection=True,
        warmup_model=False,
    )

    assert len(websocket.connect_calls) == 1
    assert len(websocket.sent) == 1
    assert websocket.sent[0]["event"] == "task_start"

    handler._last_connection_use_s = 0
    handler.maintain_connection()
    assert len(websocket.connect_calls) == 2
    assert [message["event"] for message in websocket.sent] == ["task_start", "task_start"]


def test_minimax_tts_nonstream_warms_connection_with_voice_endpoint():
    response = FakeHTTPResponse({"base_resp": {"status_code": 0, "status_msg": "success"}})
    client = FakeHTTPClient(response=response)

    handler = _minimax_handler(
        client,
        stream=False,
        warmup_connection=True,
        warmup_model=False,
    )
    handler.prewarm()

    assert len(client.calls) == 1
    url, headers, payload = client.calls[0]
    assert url == "https://api.minimax.io/v1/get_voice"
    assert headers["Accept"] == "application/json"
    assert payload == {"voice_type": "system"}
    assert response.raise_for_status_calls == 1

    handler._last_connection_use_s = 0
    handler.prewarm()
    assert len(client.calls) == 2


def test_minimax_tts_warmup_primes_model_and_exact_audio_cache(monkeypatch):
    samples = np.arange(32, dtype=np.int16)
    websocket = FakeWebSocket(
        _ws_messages(_ws_event(samples))
    )

    handler = _minimax_handler(
        FakeHTTPClient(),
        websocket=websocket,
        warmup_connection=False,
        warmup_model=True,
        model_warmup_text="Ready.",
        cache_max_mb=1,
    )
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert len(websocket.connect_calls) == 1
    assert websocket.sent[1] == {"event": "task_continue", "text": "Ready."}
    assert handler._last_model_use_s > 0

    cached = list(handler.process(TTSInput(text="Ready.", language_code="en")))

    assert b"".join(cached) == samples.tobytes()
    assert [message["event"] for message in websocket.sent] == ["task_start", "task_continue"]


def test_minimax_tts_reuses_exact_audio_from_memory_cache(monkeypatch):
    samples = np.arange(700, dtype=np.int16)
    websocket = FakeWebSocket(_ws_messages(_ws_event(samples)))
    handler = _minimax_handler(
        FakeHTTPClient(),
        websocket=websocket,
        cache_max_mb=1,
    )
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)
    tts_input = TTSInput(text="你好", language_code="zh")

    first = list(handler.process(tts_input))
    second = list(handler.process(tts_input))

    assert len(websocket.connect_calls) == 1
    assert [message["event"] for message in websocket.sent] == ["task_start", "task_continue"]
    assert b"".join(chunk.tobytes() for chunk in first) == b"".join(second)


def test_minimax_tts_yields_short_first_frame_immediately(monkeypatch):
    first = np.arange(120, dtype=np.int16)
    second = np.arange(120, 632, dtype=np.int16)
    released = Event()

    def delayed_second():
        released.wait(timeout=1)
        return _ws_event(second)

    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(first, is_final=False),
            delayed_second,
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
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

    def delayed_second():
        released.wait(timeout=1)
        return _ws_event(second)

    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(first, is_final=False),
            delayed_second,
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    gen = handler.process(TTSInput(text="hello"))
    first_chunk = next(gen)
    np.testing.assert_array_equal(first_chunk, first)
    released.set()
    rest = list(gen)
    assert len(rest) == 1
    np.testing.assert_array_equal(rest[0], second)


def test_minimax_tts_plays_audio_from_final_websocket_event(monkeypatch):
    first = np.arange(512, dtype=np.int16)
    final = np.arange(512, 1024, dtype=np.int16)
    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(first, is_final=False),
            _ws_event(final),
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], first)
    np.testing.assert_array_equal(result[1], final)


def test_minimax_tts_reassembles_split_hex_frames(monkeypatch):
    samples = np.arange(512, dtype=np.int16)
    hex_audio = _pcm_hex(samples)
    midpoint = (len(hex_audio) // 2) | 1  # odd split so a nibble is carried
    first_hex = hex_audio[:midpoint]
    second_hex = hex_audio[midpoint:]
    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(audio_hex=first_hex, is_final=False),
            _ws_event(audio_hex=second_hex),
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    played = np.concatenate(result)
    np.testing.assert_array_equal(played[: len(samples)], samples)


def test_minimax_tts_emits_end_of_response_sentinel():
    handler = _minimax_handler(FakeHTTPClient())

    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_minimax_tts_skips_punctuation_only_text(monkeypatch):
    websocket = FakeWebSocket(_ws_messages(_ws_event(np.arange(16, dtype=np.int16))))
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert list(handler.process(TTSInput(text="…"))) == []
    assert list(handler.process(TTSInput(text="..."))) == []
    assert websocket.sent == []


def test_minimax_tts_treats_empty_provider_audio_as_silent(monkeypatch):
    websocket = FakeWebSocket(
        _ws_messages(
            {
                "event": "task_continued",
                "data": {"audio": ""},
                "is_final": True,
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "extra_info": {"audio_length": 0, "word_count": 0},
            }
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    # Speakable enough to reach the provider, but MiniMax may still return silence.
    assert list(handler.process(TTSInput(text="A"))) == []
    assert [message["event"] for message in websocket.sent] == ["task_start", "task_continue"]


def test_minimax_tts_surfaces_provider_errors():
    websocket = FakeWebSocket(
        _ws_messages(
            {
                "event": "task_continued",
                "data": None,
                "is_final": True,
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
            }
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)

    with pytest.raises(RuntimeError, match="invalid api key"):
        list(handler.process(TTSInput(text="hello")))


def test_minimax_tts_drops_audio_after_interruption(monkeypatch):
    cancel_scope = CancelScope()
    websocket = FakeWebSocket(
        _ws_messages(_ws_event(np.arange(600, dtype=np.int16))),
        on_continue=cancel_scope.cancel,
    )
    handler = _minimax_handler(
        FakeHTTPClient(),
        cancel_scope=cancel_scope,
        websocket=websocket,
    )
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert list(handler.process(TTSInput(text="hello"))) == []
    assert websocket.closed is True


def test_minimax_tts_cancels_after_first_streamed_chunk(monkeypatch):
    cancel_scope = CancelScope()
    first = np.arange(512, dtype=np.int16)
    second = np.arange(512, 1024, dtype=np.int16)

    def cancel_then_second():
        cancel_scope.cancel()
        return _ws_event(second)

    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(first, is_final=False),
            cancel_then_second,
        )
    )
    handler = _minimax_handler(
        FakeHTTPClient(),
        cancel_scope=cancel_scope,
        websocket=websocket,
    )
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    result = list(handler.process(TTSInput(text="hello")))

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], first)
    assert websocket.closed is True


def test_minimax_tts_reuses_websocket_across_sentences(monkeypatch):
    websocket = FakeWebSocket(
        _ws_messages(
            _ws_event(np.arange(64, dtype=np.int16)),
            _ws_event(np.arange(64, 128, dtype=np.int16)),
        )
    )
    handler = _minimax_handler(FakeHTTPClient(), websocket=websocket)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert list(handler.process(TTSInput(text="first")))
    assert list(handler.process(TTSInput(text="second")))

    assert len(websocket.connect_calls) == 1
    assert websocket.sent[1:] == [
        {"event": "task_continue", "text": "first"},
        {"event": "task_continue", "text": "second"},
    ]


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
    websocket = FakeWebSocket(_ws_messages(_ws_event(np.arange(32, dtype=np.int16))))
    client = FakeHTTPClient()
    handler = _minimax_handler(client, websocket=websocket)
    list(handler.process(TTSInput(text="hello")))

    handler.cleanup()

    assert client.closed is False
    assert websocket.closed is True
    assert websocket.sent[-1] == {"event": "task_finish"}


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
    assert args.module_kwargs.enable_live_transcription is True
    assert args.module_kwargs.live_transcription_update_interval == 0.2
    assert args.module_kwargs.llm_backend == "chat-completions"
    assert args.module_kwargs.tts == "minimax"
    assert args.responses_api_language_model_handler_kwargs.model_name == "deepseek-v4-flash"
    assert args.responses_api_language_model_handler_kwargs.responses_api_base_url == "https://api.deepseek.com"
    assert args.responses_api_language_model_handler_kwargs.responses_api_disable_thinking is True
    assert args.responses_api_language_model_handler_kwargs.responses_api_connection_keepalive_s == 300
    assert args.responses_api_language_model_handler_kwargs.chat_size == 8
    assert args.responses_api_language_model_handler_kwargs.compact_history is False
    assert args.responses_api_language_model_handler_kwargs.stream_batch_sentences == 1
    assert args.responses_api_language_model_handler_kwargs.stream_first_chunk_lookahead_chars == 8
    assert args.responses_api_language_model_handler_kwargs.request_hedge_after_ms == 1200
    assert args.minimax_tts_handler_kwargs.minimax_tts_speed == 1.2
    assert args.vad_handler_kwargs.speculative_reopen_ms == 250
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

    def fake_setup(self, should_listen, speed=None, cancel_scope=None, speculative_turns=None):
        recorded["should_listen"] = should_listen
        recorded["speed"] = speed
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
        MiniMaxTTSHandlerArguments(minimax_tts_speed=1.25),
        cancel_scope=cancel_scope,
        speculative_turns=speculative_turns,
    )

    assert isinstance(handler, MiniMaxTTSHandler)
    assert recorded == {
        "should_listen": should_listen,
        "speed": 1.25,
        "cancel_scope": cancel_scope,
        "speculative_turns": speculative_turns,
    }
