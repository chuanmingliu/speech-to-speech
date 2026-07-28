#!/usr/bin/env python3
"""Live smoke test for the Tencent ASR, DeepSeek, and MiniMax TTS stack."""

from __future__ import annotations

import argparse
import os
import sys
import wave
from math import gcd
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import soundfile as sf
from openai import OpenAI
from scipy.signal import resample_poly

from speech_to_speech.pipeline.messages import TTSInput, VADAudio
from speech_to_speech.STT.tencent_asr_handler import TencentASRHandler
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_AUDIO = REPO_ROOT / "src" / "speech_to_speech" / "TTS" / "ref_audio.wav"
REQUIRED_ENV = (
    "DEEPSEEK_API_KEY",
    "TENCENT_ASR_SECRET_ID",
    "TENCENT_ASR_SECRET_KEY",
    "MINIMAX_TTS_API_KEY",
    "MINIMAX_TTS_VOICE_ID",
)


def _require_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {names}")


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        factor = gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // factor, sample_rate // factor)
    return np.asarray(audio, dtype=np.float32)


def smoke_tencent(audio_path: Path) -> str:
    handler = TencentASRHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
    )
    audio = _load_audio_16k_mono(audio_path)
    results = list(handler.process(VADAudio(audio=audio, mode="final")))
    if len(results) != 1:
        raise RuntimeError(f"Tencent ASR returned {len(results)} results; expected one.")
    print(f"Tencent ASR: OK ({len(audio) / 16000:.2f}s audio, {len(results[0].text)} transcript chars)")
    return results[0].text


def smoke_deepseek(prompt: str) -> str:
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "Reply with one short sentence."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("DeepSeek returned an empty response.")
    print(f"DeepSeek: OK ({len(text)} response chars)")
    return text


def smoke_minimax(text: str, output_path: Path) -> int:
    handler = MiniMaxTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
    )
    try:
        chunks = list(handler.process(TTSInput(text=text, language_code="zh")))
    finally:
        handler.cleanup()
    if not chunks:
        raise RuntimeError("MiniMax returned no audio chunks.")

    audio = np.concatenate(chunks).astype("<i2", copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio.tobytes())
    print(f"MiniMax TTS: OK ({len(audio) / 16000:.2f}s audio, saved to {output_path})")
    return len(audio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-audio", type=Path, default=DEFAULT_ASR_AUDIO)
    parser.add_argument("--llm-prompt", default="Say that the DeepSeek smoke test passed.")
    parser.add_argument("--tts-text", default="你好，这是 MiniMax 语音合成测试。")
    parser.add_argument(
        "--tts-output",
        type=Path,
        default=Path("/tmp/custom-services-minimax-smoke.wav"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_environment()
    failures: list[tuple[str, Exception]] = []

    for name, operation in (
        ("Tencent ASR", lambda: smoke_tencent(args.asr_audio)),
        ("DeepSeek", lambda: smoke_deepseek(args.llm_prompt)),
        ("MiniMax TTS", lambda: smoke_minimax(args.tts_text, args.tts_output)),
    ):
        try:
            operation()
        except Exception as exc:
            failures.append((name, exc))
            print(f"{name}: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(f"Smoke test failed for {len(failures)} provider(s).", file=sys.stderr)
        return 1
    print("All custom-service smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
