from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from threading import Event, Thread
from time import monotonic

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.tencent_realtime_client import (
    TencentRealtimeConfig,
    TencentRealtimeSession,
    TencentRecognitionResult,
)

logger = logging.getLogger(__name__)


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
        clock: Callable[[], float] = monotonic,
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
        self._clock = clock
        self._active_key: tuple[str | None, int | None] | None = None
        self._active_session: TencentRealtimeSession | None = None
        self._speech_started_at_s: float | None = None
        self._first_partial_at_s: float | None = None
        self._final_at_s: float | None = None
        self._checkpoint_turn_id: str | None = None
        self._checkpoint_revision: int | None = None
        self._checkpoint_samples = 0
        self._checkpoint_text = ""

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        key = (vad_audio.turn_id, vad_audio.turn_revision)
        session = self._session_for(key, vad_audio.created_at_s)
        provider_audio = self._provider_audio(vad_audio)
        try:
            if vad_audio.mode == "progressive":
                session.push_snapshot(provider_audio)
            else:
                finish_done = Event()
                stale_watcher: Thread | None = None
                if self.speculative_turns is not None and vad_audio.turn_id is not None:

                    def abort_if_stale() -> None:
                        while not finish_done.wait(0.01):
                            if not self.speculative_turns.is_latest(
                                vad_audio.turn_id,
                                vad_audio.turn_revision,
                            ):
                                session.close()
                                logger.info(
                                    "Tencent finalization cancelled for stale turn=%s rev=%s",
                                    vad_audio.turn_id,
                                    vad_audio.turn_revision,
                                )
                                return

                    stale_watcher = Thread(
                        target=abort_if_stale,
                        name="tencent-asr-stale-finalization",
                        daemon=True,
                    )
                    stale_watcher.start()
                try:
                    session.finish(provider_audio)
                finally:
                    finish_done.set()
                    if stale_watcher is not None:
                        stale_watcher.join(0.05)

            for result in session.drain_results():
                yield self._message_for(result, vad_audio)
        except Exception:
            self._close_active()
            raise
        finally:
            if vad_audio.mode != "progressive":
                self._close_active()

    def _session_for(
        self,
        key: tuple[str | None, int | None],
        speech_started_at_s: float,
    ) -> TencentRealtimeSession:
        if self._active_session is not None and self._active_key != key:
            self._close_active()
        if self._active_session is None:
            self._active_session = self._session_factory(self._config)
            self._active_key = key
            self._speech_started_at_s = speech_started_at_s
            self._first_partial_at_s = None
            self._final_at_s = None
        return self._active_session

    def _message_for(self, result: TencentRecognitionResult, vad_audio: STTIn) -> STTOut:
        text = f"{self._checkpoint_prefix(vad_audio)}{result.text}"
        if result.final:
            if self._final_at_s is None:
                self._final_at_s = self._clock()
                logger.info(
                    "Tencent final latency: %.3fs (turn=%s rev=%s)",
                    self._final_at_s - vad_audio.created_at_s,
                    vad_audio.turn_id,
                    vad_audio.turn_revision,
                )
            else:
                logger.debug(
                    "Tencent final latency: %.3fs (turn=%s rev=%s)",
                    self._final_at_s - vad_audio.created_at_s,
                    vad_audio.turn_id,
                    vad_audio.turn_revision,
                )
            final_text = text.strip()
            if self.speculative_turns is None or self.speculative_turns.is_latest(
                vad_audio.turn_id,
                vad_audio.turn_revision,
            ):
                self._checkpoint_turn_id = vad_audio.turn_id
                self._checkpoint_revision = vad_audio.turn_revision
                self._checkpoint_samples = len(vad_audio.audio)
                self._checkpoint_text = final_text
            return Transcription(
                text=final_text,
                language_code=self.language_code,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
                final_at_s=self._final_at_s,
            )
        if self._first_partial_at_s is None:
            self._first_partial_at_s = self._clock()
            log = logger.info
        else:
            log = logger.debug
        speech_started_at_s = (
            self._speech_started_at_s if self._speech_started_at_s is not None else vad_audio.created_at_s
        )
        log(
            "Tencent first partial latency: %.3fs (turn=%s rev=%s)",
            self._first_partial_at_s - speech_started_at_s,
            vad_audio.turn_id,
            vad_audio.turn_revision,
        )
        return PartialTranscription(
            text=text,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            first_partial_at_s=self._first_partial_at_s,
        )

    def _checkpoint_prefix(self, vad_audio: STTIn) -> str:
        if self._has_checkpoint_for(vad_audio):
            return self._checkpoint_text
        return ""

    def _has_checkpoint_for(self, vad_audio: STTIn) -> bool:
        return (
            vad_audio.turn_id == self._checkpoint_turn_id
            and self._checkpoint_revision is not None
            and vad_audio.turn_revision is not None
            and vad_audio.turn_revision > self._checkpoint_revision
        )

    def _provider_audio(self, vad_audio: STTIn):
        if self._has_checkpoint_for(vad_audio):
            return vad_audio.audio[self._checkpoint_samples :]
        return vad_audio.audio

    def _close_active(self) -> None:
        session, self._active_session = self._active_session, None
        self._active_key = None
        self._speech_started_at_s = None
        self._first_partial_at_s = None
        self._final_at_s = None
        if session is not None:
            session.close()

    def cleanup(self) -> None:
        self._close_active()

    def on_session_end(self) -> None:
        self._close_active()
        self._checkpoint_turn_id = None
        self._checkpoint_revision = None
        self._checkpoint_samples = 0
        self._checkpoint_text = ""
        super().on_session_end()
