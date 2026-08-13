import io
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
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.s2s_pipeline import get_stt_handler, get_tts_handler, parse_arguments
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler


class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body
        self.raise_for_status_calls = 0

    def raise_for_status(self):
        self.raise_for_status_calls += 1

    def json(self):
        return self.body


class FakeHTTPClient:
    def __init__(self, response, on_post=None):
        self.response = response
        self.on_post = on_post
        self.calls = []
        self.closed = False

    def post(self, url, *, headers, json):
        self.calls.append((url, headers, json))
        if self.on_post:
            self.on_post()
        return self.response

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


def _minimax_response(samples):
    return FakeHTTPResponse(
        {
            "data": {"audio": _wav_hex(samples), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
    )


def _minimax_handler(client, cancel_scope=None):
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
        },
    )


def test_minimax_tts_sends_expected_payload_and_yields_padded_pcm(monkeypatch):
    samples = np.arange(700, dtype=np.int16)
    response = _minimax_response(samples)
    client = FakeHTTPClient(response)
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
    assert payload["model"] == "speech-2.8-turbo"
    assert payload["text"] == "你好"
    assert payload["stream"] is False
    assert payload["output_format"] == "hex"
    assert payload["voice_setting"]["voice_id"] == "test-voice"
    assert payload["audio_setting"] == {
        "sample_rate": 16000,
        "format": "wav",
        "channel": 1,
    }
    assert response.raise_for_status_calls == 1


def test_minimax_tts_emits_end_of_response_sentinel():
    handler = _minimax_handler(FakeHTTPClient(_minimax_response([])))

    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_minimax_tts_surfaces_provider_errors():
    response = FakeHTTPResponse(
        {
            "data": None,
            "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
        }
    )
    handler = _minimax_handler(FakeHTTPClient(response))

    with pytest.raises(RuntimeError, match="invalid api key"):
        list(handler.process(TTSInput(text="hello")))


def test_minimax_tts_drops_audio_after_interruption(monkeypatch):
    cancel_scope = CancelScope()
    client = FakeHTTPClient(_minimax_response(np.arange(600, dtype=np.int16)), on_post=cancel_scope.cancel)
    handler = _minimax_handler(client, cancel_scope=cancel_scope)
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    assert list(handler.process(TTSInput(text="hello"))) == []


def test_minimax_tts_validates_returned_sample_rate(monkeypatch):
    response = FakeHTTPResponse(
        {
            "data": {"audio": _wav_hex(np.zeros(10, dtype=np.int16), sample_rate=24000)},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
    )
    handler = _minimax_handler(FakeHTTPClient(response))
    monkeypatch.setattr("speech_to_speech.TTS.minimax_tts_handler.console.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="24000 Hz"):
        list(handler.process(TTSInput(text="hello")))


def test_minimax_cleanup_does_not_close_injected_client():
    client = FakeHTTPClient(_minimax_response([]))
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
    assert args.module_kwargs.enable_live_transcription is True
    assert args.module_kwargs.live_transcription_update_interval == 0.2
    assert args.module_kwargs.llm_backend == "chat-completions"
    assert args.module_kwargs.tts == "minimax"
    assert args.responses_api_language_model_handler_kwargs.model_name == "deepseek-v4-flash"
    assert args.responses_api_language_model_handler_kwargs.responses_api_base_url == "https://api.deepseek.com"
    assert args.responses_api_language_model_handler_kwargs.responses_api_disable_thinking is True


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
