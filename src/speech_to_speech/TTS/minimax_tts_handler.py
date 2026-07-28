from __future__ import annotations

import io
import logging
import os
import wave
from threading import Event
from typing import Any, Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


class MiniMaxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """MiniMax synchronous T2A adapter producing 16 kHz mono PCM16 chunks."""

    def setup(
        self,
        should_listen: Event,
        api_key: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        endpoint: str | None = None,
        language_boost: str | None = None,
        sample_rate: int = 16000,
        blocksize: int = 512,
        request_timeout_s: float = 30.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        client: Any | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.api_key = api_key or os.getenv("MINIMAX_TTS_API_KEY")
        self.model = model or os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-turbo")
        self.voice_id = voice_id or os.getenv("MINIMAX_TTS_VOICE_ID")
        self.endpoint = endpoint or os.getenv(
            "MINIMAX_TTS_ENDPOINT",
            "https://api.minimax.io/v1/t2a_v2",
        )
        self.language_boost = language_boost or os.getenv("MINIMAX_TTS_LANGUAGE_BOOST", "auto")
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        if not self.api_key:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_API_KEY.")
        if not self.voice_id:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_VOICE_ID.")

        self.client = client or httpx.Client(timeout=request_timeout_s)
        self._owns_client = client is None

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": text,
            "stream": False,
            "language_boost": self.language_boost,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "format": "wav",
                "channel": 1,
            },
        }

    def _decode_wav(self, audio_hex: str) -> np.ndarray:
        try:
            wav_bytes = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
        except wave.Error as exc:
            raise ValueError("MiniMax returned an invalid WAV payload.") from exc

        if channels != 1:
            raise ValueError(f"MiniMax returned {channels} audio channels; expected mono.")
        if sample_width != 2:
            raise ValueError(f"MiniMax returned {sample_width * 8}-bit audio; expected signed PCM16.")
        if sample_rate != self.sample_rate:
            raise ValueError(f"MiniMax returned {sample_rate} Hz audio; expected {self.sample_rate} Hz.")
        return np.frombuffer(frames, dtype="<i2")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale MiniMax TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        generation = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text.strip()
        if not text:
            return
        console.print(f"[green]ASSISTANT: {text}")

        response = self.client.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self._payload(text),
        )
        response.raise_for_status()
        body = response.json()
        base_response = body.get("base_resp") or {}
        status_code = base_response.get("status_code")
        if status_code != 0:
            raise RuntimeError(
                "MiniMax TTS request failed "
                f"(status_code={status_code!r}): {base_response.get('status_msg', 'unknown error')}"
            )

        data = body.get("data")
        if not data or not data.get("audio"):
            raise RuntimeError("MiniMax TTS response did not contain audio data.")
        audio = self._decode_wav(data["audio"])

        for start in range(0, len(audio), self.blocksize):
            if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                logger.info("MiniMax TTS playback cancelled (interruption)")
                return
            chunk = audio[start : start + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield np.asarray(chunk, dtype=np.int16)

    def cleanup(self) -> None:
        if self._owns_client:
            self.client.close()
