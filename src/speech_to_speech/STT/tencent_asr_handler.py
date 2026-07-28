from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

TENCENT_SENTENCE_RECOGNITION_MAX_SECONDS = 60.0


class TencentASRHandler(BaseSTTHandler):
    """Tencent Cloud SentenceRecognition adapter for VAD-finalized utterances."""

    def setup(
        self,
        secret_id: str | None = None,
        secret_key: str | None = None,
        engine: str | None = None,
        language_code: str | None = None,
        endpoint: str = "asr.tencentcloudapi.com",
        client: Any | None = None,
    ) -> None:
        self.engine = engine or os.getenv("TENCENT_ASR_ENGINE", "16k_zh")
        self.language_code = language_code or os.getenv("TENCENT_ASR_LANGUAGE", "zh")
        self.sample_rate = 16000

        if client is not None:
            self.client = client
            self._request_model_type: type[Any] | None = None
            return

        secret_id = secret_id or os.getenv("TENCENT_ASR_SECRET_ID")
        secret_key = secret_key or os.getenv("TENCENT_ASR_SECRET_KEY")
        if not secret_id or not secret_key:
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
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        cred = credential.Credential(secret_id, secret_key)
        self.client = asr_client.AsrClient(cred, "", client_profile)
        self._request_model_type = models.SentenceRecognitionRequest

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

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        # SentenceRecognition is a finalized-utterance API. Progressive VAD
        # snapshots are intentionally ignored instead of generating billable,
        # duplicate requests.
        if vad_audio.mode == "progressive":
            return

        audio = np.asarray(vad_audio.audio).reshape(-1)
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
        response = self.client.SentenceRecognition(self._new_request(payload))
        text = (response.Result if hasattr(response, "Result") else response["Result"]) or ""
        text = text.strip()

        if text:
            console.print(f"[yellow]USER: {text}")

        yield Transcription(
            text=text,
            language_code=self.language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
