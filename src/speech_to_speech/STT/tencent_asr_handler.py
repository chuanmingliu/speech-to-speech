from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.tencent_realtime import PCM_FRAME_BYTES, TencentRealtimeASRSession, build_realtime_url

logger = logging.getLogger(__name__)
console = Console()

TENCENT_SENTENCE_RECOGNITION_MAX_SECONDS = 60.0
# Reuse a silence-prefetch result when the finalized utterance only added a
# short tail (typical VAD min_silence plus a couple of frames).
TENCENT_SPECULATIVE_REUSE_EXTRA_S = 0.35


class TencentASRHandler(BaseSTTHandler):
    """Tencent Cloud ASR adapter.

    The Tencent profile streams progressive VAD audio over the realtime
    WebSocket API when ``TENCENT_ASR_APP_ID`` is set, so recognition overlaps
    speech and emits ``PartialTranscription``. Without an App ID,
    SentenceRecognition runs on finalized (or silence-prefetched) audio.
    """

    def setup(
        self,
        secret_id: str | None = None,
        secret_key: str | None = None,
        engine: str | None = None,
        language_code: str | None = None,
        endpoint: str = "asr.tencentcloudapi.com",
        app_id: str | None = None,
        client: Any | None = None,
        realtime_session_factory: Callable[[], TencentRealtimeASRSession] | None = None,
    ) -> None:
        self.engine = engine or os.getenv("TENCENT_ASR_ENGINE", "16k_zh")
        self.language_code = language_code or os.getenv("TENCENT_ASR_LANGUAGE", "zh")
        self.sample_rate = 16000
        self.secret_id = secret_id or os.getenv("TENCENT_ASR_SECRET_ID")
        self.secret_key = secret_key or os.getenv("TENCENT_ASR_SECRET_KEY")
        self.app_id = (app_id or os.getenv("TENCENT_ASR_APP_ID") or "").strip()
        self._executor: ThreadPoolExecutor | None = None
        self._speculative: dict[str, Any] | None = None
        self._realtime_factory = realtime_session_factory
        self._rt_session: TencentRealtimeASRSession | None = None
        self._rt_key: tuple[str | None, int | None] | None = None
        self._rt_sent = 0
        self._last_partial = ""
        self._use_realtime = realtime_session_factory is not None or bool(self.app_id)

        if client is not None:
            self.client = client
            self._request_model_type: type[Any] | None = None
            # Injected clients are the SentenceRecognition fake unless a
            # realtime factory is also provided.
            self._use_realtime = realtime_session_factory is not None
            return

        if not self.secret_id or not self.secret_key:
            raise ValueError("Tencent ASR requires TENCENT_ASR_SECRET_ID and TENCENT_ASR_SECRET_KEY.")

        try:
            from tencentcloud.asr.v20190614 import asr_client, models
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Tencent ASR requires the optional dependency. Install it with "
                '`pip install "speech-to-speech[tencent-asr]"`.'
            ) from exc

        http_profile = HttpProfile()
        http_profile.endpoint = endpoint
        http_profile.reqTimeout = int(os.getenv("TENCENT_ASR_TIMEOUT_S", "10"))
        if hasattr(http_profile, "keepAlive"):
            http_profile.keepAlive = True
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        cred = credential.Credential(self.secret_id, self.secret_key)
        self.client = asr_client.AsrClient(cred, "", client_profile)
        self._request_model_type = models.SentenceRecognitionRequest
        if self._use_realtime:
            logger.info("Tencent ASR realtime WebSocket enabled")
        else:
            logger.warning(
                "Tencent ASR using SentenceRecognition (non-streaming); "
                "set TENCENT_ASR_APP_ID to stream PCM over the realtime WebSocket"
            )

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tencent-asr")
        return self._executor

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        audio_float = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio_float)):
            raise ValueError("Tencent ASR input contains non-finite audio samples.")
        return (np.clip(audio_float, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def _new_request(self, payload: dict[str, Any]) -> Any:
        if self._request_model_type is None:
            return payload
        request = self._request_model_type()
        request.from_json_string(json.dumps(payload))
        return request

    def _recognize(self, audio: np.ndarray, *, turn_id: str | None, turn_revision: int | None) -> str:
        duration_s = len(audio) / self.sample_rate
        if duration_s > TENCENT_SENTENCE_RECOGNITION_MAX_SECONDS:
            raise ValueError(
                "Tencent SentenceRecognition accepts at most 60 seconds per utterance; "
                f"received {duration_s:.2f} seconds."
            )
        pcm = self._to_pcm16(audio)
        payload = {
            "EngSerViceType": self.engine,
            "SourceType": 1,
            "VoiceFormat": "pcm",
            "Data": base64.b64encode(pcm).decode("ascii"),
            "DataLen": len(pcm),
        }
        started_at_s = perf_counter()
        response = self.client.SentenceRecognition(self._new_request(payload))
        logger.info(
            "Tencent ASR completed in %.3fs (audio=%.2fs, turn=%s rev=%s)",
            perf_counter() - started_at_s,
            duration_s,
            turn_id,
            turn_revision,
        )
        text = (response.Result if hasattr(response, "Result") else response["Result"]) or ""
        return text.strip()

    def _start_speculative(self, vad_audio: VADAudio) -> None:
        audio = np.asarray(vad_audio.audio).reshape(-1)
        if audio.size < int(0.4 * self.sample_rate):
            return
        in_flight = self._speculative
        if (
            in_flight is not None
            and in_flight["turn_id"] == vad_audio.turn_id
            and in_flight["turn_revision"] == vad_audio.turn_revision
            and not in_flight["future"].done()
        ):
            return
        future = self._pool().submit(
            self._recognize,
            audio,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )
        self._speculative = {
            "turn_id": vad_audio.turn_id,
            "turn_revision": vad_audio.turn_revision,
            "n_samples": int(audio.size),
            "future": future,
        }
        logger.info(
            "Tencent ASR speculative request started (audio=%.2fs, turn=%s rev=%s)",
            audio.size / self.sample_rate,
            vad_audio.turn_id,
            vad_audio.turn_revision,
        )

    def _take_speculative(self, vad_audio: VADAudio) -> Future[str] | None:
        spec = self._speculative
        self._speculative = None
        if spec is None:
            return None
        if spec["turn_id"] != vad_audio.turn_id or spec["turn_revision"] != vad_audio.turn_revision:
            return None
        extra_s = (len(np.asarray(vad_audio.audio).reshape(-1)) - spec["n_samples"]) / self.sample_rate
        if extra_s > TENCENT_SPECULATIVE_REUSE_EXTRA_S:
            logger.info(
                "Tencent ASR speculative result discarded (extra=%.3fs, turn=%s rev=%s)",
                extra_s,
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            return None
        return spec["future"]

    def _new_realtime_session(self) -> TencentRealtimeASRSession:
        if self._realtime_factory is not None:
            return self._realtime_factory()
        if not self.app_id or not self.secret_id or not self.secret_key:
            raise RuntimeError("Tencent realtime ASR requires TENCENT_ASR_APP_ID and API keys.")
        url = build_realtime_url(
            app_id=self.app_id,
            secret_id=self.secret_id,
            secret_key=self.secret_key,
            engine=self.engine,
        )
        return TencentRealtimeASRSession(url)

    def _ensure_realtime_session(self, vad_audio: VADAudio) -> TencentRealtimeASRSession:
        key = (vad_audio.turn_id, vad_audio.turn_revision)
        if self._rt_session is not None and self._rt_key == key:
            return self._rt_session
        self._close_realtime()
        session = self._new_realtime_session()
        session.start()
        self._rt_session = session
        self._rt_key = key
        self._rt_sent = 0
        logger.info(
            "Tencent realtime ASR session started (turn=%s rev=%s)",
            vad_audio.turn_id,
            vad_audio.turn_revision,
        )
        return session

    def _send_realtime_audio(self, audio: np.ndarray) -> str:
        if self._rt_session is None:
            return ""
        if audio.size <= self._rt_sent:
            return self._rt_session.current_text()
        new_audio = audio[self._rt_sent :]
        self._rt_sent = int(audio.size)
        pcm = self._to_pcm16(new_audio)
        text = ""
        for start in range(0, len(pcm), PCM_FRAME_BYTES):
            text = self._rt_session.send_pcm(pcm[start : start + PCM_FRAME_BYTES])
        return text

    def _close_realtime(self) -> None:
        if self._rt_session is None:
            return
        self._rt_session.close()
        self._rt_session = None
        self._rt_key = None
        self._rt_sent = 0
        self._last_partial = ""

    def _process_realtime(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if not isinstance(vad_audio, VADAudio):
            return
        audio = np.asarray(vad_audio.audio).reshape(-1)
        if vad_audio.mode == "progressive":
            self._ensure_realtime_session(vad_audio)
            text = self._send_realtime_audio(audio)
            if text:
                if text != self._last_partial:
                    logger.info(
                        "Streaming ASR partial (turn=%s rev=%s): %s",
                        vad_audio.turn_id,
                        vad_audio.turn_revision,
                        text if len(text) <= 80 else f"{text[:80]}…",
                    )
                    self._last_partial = text
                yield PartialTranscription(
                    text=text,
                    turn_id=vad_audio.turn_id,
                    turn_revision=vad_audio.turn_revision,
                )
            return

        started_at_s = perf_counter()
        try:
            self._ensure_realtime_session(vad_audio)
            self._send_realtime_audio(audio)
            text = self._rt_session.finish() if self._rt_session is not None else ""
        finally:
            self._close_realtime()
        logger.info(
            "Streaming ASR final in %.3fs (audio=%.2fs, turn=%s rev=%s): %s",
            perf_counter() - started_at_s,
            audio.size / self.sample_rate,
            vad_audio.turn_id,
            vad_audio.turn_revision,
            text if len(text) <= 80 else f"{text[:80]}…",
        )
        if text:
            console.print(f"[yellow]USER: {text}")
        yield Transcription(
            text=text,
            language_code=self.language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def _process_sentence(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            if isinstance(vad_audio, VADAudio):
                self._start_speculative(vad_audio)
            return

        audio = np.asarray(vad_audio.audio).reshape(-1)
        started_at_s = perf_counter()
        text = ""
        reused = False
        speculative = self._take_speculative(vad_audio) if isinstance(vad_audio, VADAudio) else None
        if speculative is not None:
            text = speculative.result(timeout=float(os.getenv("TENCENT_ASR_TIMEOUT_S", "10")))
            reused = True
            logger.info(
                "Tencent ASR reused speculative result in %.3fs (turn=%s rev=%s)",
                perf_counter() - started_at_s,
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
        else:
            text = self._recognize(
                audio,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )

        if text:
            console.print(f"[yellow]USER: {text}")
        elif reused:
            logger.debug("Tencent ASR speculative transcript was empty")

        yield Transcription(
            text=text,
            language_code=self.language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if self._use_realtime:
            try:
                yield from self._process_realtime(vad_audio)
                return
            except Exception:
                self._close_realtime()
                if vad_audio.mode == "progressive":
                    logger.exception("Tencent realtime ASR progressive update failed")
                    return
                logger.exception("Tencent realtime ASR failed; falling back to SentenceRecognition")
        yield from self._process_sentence(vad_audio)

    def cleanup(self) -> None:
        self._close_realtime()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._speculative = None

    def on_session_end(self) -> None:
        self._close_realtime()
        self._speculative = None
        super().on_session_end()
