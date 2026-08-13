from __future__ import annotations

import os
from collections.abc import Callable, Iterator

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.tencent_realtime_client import (
    TencentRealtimeConfig,
    TencentRealtimeSession,
    TencentRecognitionResult,
)


class TencentASRHandler(BaseSTTHandler):
    """Tencent realtime ASR adapter with one provider session per speech turn."""

    def setup(
        self,
        app_id: str | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        engine: str | None = None,
        language_code: str | None = None,
        endpoint: str = "asr.cloud.tencent.com",
        session_factory: Callable[[TencentRealtimeConfig], TencentRealtimeSession] = TencentRealtimeSession,
    ) -> None:
        self.language_code = language_code or os.getenv("TENCENT_ASR_LANGUAGE", "zh")
        self._config = TencentRealtimeConfig(
            app_id=app_id or os.getenv("TENCENT_ASR_APP_ID", ""),
            secret_id=secret_id or os.getenv("TENCENT_ASR_SECRET_ID", ""),
            secret_key=secret_key or os.getenv("TENCENT_ASR_SECRET_KEY", ""),
            engine=engine or os.getenv("TENCENT_ASR_ENGINE", "16k_zh"),
            endpoint=endpoint,
        )
        self._session_factory = session_factory
        self._active_key: tuple[str | None, int | None] | None = None
        self._active_session: TencentRealtimeSession | None = None

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        key = (vad_audio.turn_id, vad_audio.turn_revision)
        session = self._session_for(key)
        try:
            if vad_audio.mode == "progressive":
                session.push_snapshot(vad_audio.audio)
            else:
                session.finish(vad_audio.audio)

            for result in session.drain_results():
                yield self._message_for(result, vad_audio)
        except Exception:
            self._close_active()
            raise
        finally:
            if vad_audio.mode != "progressive":
                self._close_active()

    def _session_for(self, key: tuple[str | None, int | None]) -> TencentRealtimeSession:
        if self._active_session is not None and self._active_key != key:
            self._close_active()
        if self._active_session is None:
            self._active_session = self._session_factory(self._config)
            self._active_key = key
        return self._active_session

    def _message_for(self, result: TencentRecognitionResult, vad_audio: STTIn) -> STTOut:
        if result.final:
            return Transcription(
                text=result.text.strip(),
                language_code=self.language_code,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        return PartialTranscription(
            text=result.text,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )

    def _close_active(self) -> None:
        session, self._active_session = self._active_session, None
        self._active_key = None
        if session is not None:
            session.close()

    def cleanup(self) -> None:
        self._close_active()

    def on_session_end(self) -> None:
        self._close_active()
        super().on_session_end()
