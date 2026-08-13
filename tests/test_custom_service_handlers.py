import sys
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

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
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.s2s_pipeline import get_stt_handler, get_tts_handler, parse_arguments
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler


def test_minimax_tts_emits_end_of_response_sentinel():
    handler = MiniMaxTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"api_key": "test-key", "voice_id": "test-voice"},
    )

    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


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
