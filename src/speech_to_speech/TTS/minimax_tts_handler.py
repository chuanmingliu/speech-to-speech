from __future__ import annotations

import io
import json
import logging
import os
import wave
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class MiniMaxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """MiniMax T2A adapter producing 16 kHz mono PCM16 chunks.

    Streaming is on by default so the first hex PCM frame can be played before
    the provider finishes the utterance. Set ``MINIMAX_TTS_STREAM=false`` to
    fall back to a single hex-encoded WAV response.
    """

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
        stream: bool | None = None,
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
        self.stream = _env_flag("MINIMAX_TTS_STREAM", True) if stream is None else stream

        if not self.api_key:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_API_KEY.")
        if not self.voice_id:
            raise ValueError("MiniMax TTS requires MINIMAX_TTS_VOICE_ID.")

        self.client = client or httpx.Client(
            timeout=httpx.Timeout(request_timeout_s, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=30.0),
        )
        self._owns_client = client is None
        if self._owns_client:
            self._warmup_connection()

    def _warmup_connection(self) -> None:
        try:
            parsed = httpx.URL(self.endpoint)
            origin = f"{parsed.scheme}://{parsed.host}"
            self.client.head(origin, timeout=3.0)
        except Exception as exc:
            logger.debug("MiniMax connection warmup skipped: %s", exc)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.stream else "application/json",
        }

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": text,
            "stream": self.stream,
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
                "format": "pcm" if self.stream else "wav",
                "channel": 1,
            },
        }

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    @staticmethod
    def _raise_if_failed(body: dict[str, Any]) -> None:
        base_response = body.get("base_resp") or {}
        status_code = base_response.get("status_code")
        if status_code not in (None, 0):
            raise RuntimeError(
                "MiniMax TTS request failed "
                f"(status_code={status_code!r}): {base_response.get('status_msg', 'unknown error')}"
            )

    def _decode_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
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
        return np.frombuffer(frames, dtype="<i2").copy()

    def _decode_wav(self, audio_hex: str) -> np.ndarray:
        try:
            wav_bytes = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        return self._decode_wav_bytes(wav_bytes)

    def _decode_audio_payload(self, audio_hex: str) -> np.ndarray:
        try:
            raw = bytes.fromhex(audio_hex)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        if raw.startswith(b"RIFF"):
            return self._decode_wav_bytes(raw)
        if len(raw) % 2:
            raw = raw[:-1]
        if not raw:
            return np.array([], dtype=np.int16)
        return np.frombuffer(raw, dtype="<i2").copy()

    def _decode_hex_pcm(self, audio_hex: str, leftover_hex: str = "") -> tuple[np.ndarray, str]:
        hex_str = leftover_hex + (audio_hex or "")
        if not hex_str:
            return np.array([], dtype=np.int16), ""
        if len(hex_str) % 2:
            leftover_hex = hex_str[-1]
            hex_str = hex_str[:-1]
        else:
            leftover_hex = ""
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError as exc:
            raise ValueError("MiniMax returned invalid hex-encoded audio.") from exc
        if len(raw) % 2:
            leftover_hex = f"{raw[-1]:02x}" + leftover_hex
            raw = raw[:-1]
        if not raw:
            return np.array([], dtype=np.int16), leftover_hex
        return np.frombuffer(raw, dtype="<i2").copy(), leftover_hex

    @staticmethod
    def _parse_sse_event(raw: str) -> dict[str, Any] | None:
        data_lines: list[str] = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return None
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("MiniMax returned an invalid streaming event.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("MiniMax streaming event must be a JSON object.")
        return parsed

    def _iter_sse_json(self, response: Any) -> Iterator[dict[str, Any]]:
        iter_text = getattr(response, "iter_text", None)
        if iter_text is None:
            raise RuntimeError("MiniMax streaming client does not support iter_text().")
        buffer = ""
        for chunk in iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                event = self._parse_sse_event(raw)
                if event is not None:
                    yield event
        if buffer.strip():
            event = self._parse_sse_event(buffer)
            if event is not None:
                yield event

    def _emit_pcm(self, samples: np.ndarray, generation: int | None, *, pad: bool) -> Iterator[np.ndarray]:
        if samples.size == 0:
            return
        for start in range(0, len(samples), self.blocksize):
            if self._is_cancelled(generation):
                logger.info("MiniMax TTS playback cancelled (interruption)")
                return
            chunk = samples[start : start + self.blocksize]
            if len(chunk) < self.blocksize:
                if not pad:
                    return
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield np.asarray(chunk, dtype=np.int16)

    def _log_first_audio_latency(self, tts_input: TTSInput, started_at_s: float) -> None:
        request_s = perf_counter() - started_at_s
        logger.info(
            "MiniMax TTS first audio in %.3fs (turn=%s rev=%s)",
            request_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )
        if tts_input.speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - tts_input.speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _synthesize_streaming(
        self,
        text: str,
        generation: int | None,
        tts_input: TTSInput,
    ) -> Iterator[np.ndarray]:
        started_at_s = perf_counter()
        with self.client.stream(
            "POST",
            self.endpoint,
            headers=self._headers(),
            json=self._payload(text),
        ) as response:
            response.raise_for_status()
            leftover_hex = ""
            pending = np.array([], dtype=np.int16)
            got_incremental = False
            final_audio_hex: str | None = None
            first_audio = True

            for event in self._iter_sse_json(response):
                if self._is_cancelled(generation):
                    logger.info("MiniMax TTS playback cancelled (interruption)")
                    return
                self._raise_if_failed(event)
                data = event.get("data") or {}
                audio_hex = data.get("audio")
                status = data.get("status")
                if not audio_hex:
                    continue
                if status == 2:
                    # Aggregated copy of the whole utterance. Skip while
                    # incremental frames are already playing.
                    final_audio_hex = audio_hex
                    continue

                got_incremental = True
                samples, leftover_hex = self._decode_hex_pcm(audio_hex, leftover_hex)
                if samples.size == 0:
                    continue
                pending = np.concatenate((pending, samples)) if pending.size else samples
                if first_audio:
                    self._log_first_audio_latency(tts_input, started_at_s)
                    first_audio = False
                    # Do not wait for a full playback block before the first
                    # samples leave the handler.
                    if len(pending) < self.blocksize:
                        yield np.asarray(pending, dtype=np.int16)
                        pending = np.array([], dtype=np.int16)
                        continue
                emit_upto = len(pending) - (len(pending) % self.blocksize)
                if emit_upto:
                    yield from self._emit_pcm(pending[:emit_upto], generation, pad=False)
                    pending = pending[emit_upto:]
                    if self._is_cancelled(generation):
                        return

            if self._is_cancelled(generation):
                return
            if not got_incremental:
                if not final_audio_hex:
                    raise RuntimeError("MiniMax TTS response did not contain audio data.")
                pending = self._decode_audio_payload(final_audio_hex)
                if first_audio and pending.size:
                    self._log_first_audio_latency(tts_input, started_at_s)
            yield from self._emit_pcm(pending, generation, pad=True)

    def _synthesize_sync(
        self,
        text: str,
        generation: int | None,
        tts_input: TTSInput,
    ) -> Iterator[np.ndarray]:
        started_at_s = perf_counter()
        response = self.client.post(
            self.endpoint,
            headers=self._headers(),
            json=self._payload(text),
        )
        response.raise_for_status()
        body = response.json()
        self._raise_if_failed(body)

        data = body.get("data")
        if not data or not data.get("audio"):
            raise RuntimeError("MiniMax TTS response did not contain audio data.")
        audio = self._decode_wav(data["audio"])
        if audio.size:
            self._log_first_audio_latency(tts_input, started_at_s)
        yield from self._emit_pcm(audio, generation, pad=True)

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

        if self.stream:
            yield from self._synthesize_streaming(text, generation, tts_input)
        else:
            yield from self._synthesize_sync(text, generation, tts_input)

    def cleanup(self) -> None:
        if self._owns_client:
            self.client.close()
