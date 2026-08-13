from __future__ import annotations

from typing import Any

import av
import numpy as np


class IncrementalMP3Decoder:
    """Incrementally decode MP3 bytes into fixed-size mono PCM16 blocks."""

    max_fragment_bytes = 1024 * 1024

    def __init__(self, sample_rate: int = 16_000, channels: int = 1, block_samples: int = 512) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels != 1:
            raise ValueError("only mono output is supported")
        if block_samples <= 0:
            raise ValueError("block_samples must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_samples = block_samples
        self._reservoir_limit = block_samples * 2
        self._reservoir = np.empty(0, dtype=np.int16)
        self._codec: Any | None = av.CodecContext.create("mp3", "r")
        self._resampler: Any | None = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        self._finished = False

    def feed(self, encoded: bytes) -> list[np.ndarray]:
        if self._finished or self._codec is None:
            raise RuntimeError("decoder is closed")
        if len(encoded) > self.max_fragment_bytes:
            raise ValueError("MP3 fragment exceeds 1 MiB")
        return self._decode_packets(self._codec.parse(encoded))

    def finish(self) -> list[np.ndarray]:
        if self._finished:
            return []
        if self._codec is None or self._resampler is None:
            self._finished = True
            return []

        blocks = self._decode_packets(self._codec.parse(b""))
        for frame in self._codec.decode(None):
            blocks.extend(self._consume_decoded_frame(frame))
        for frame in self._resampler.resample(None):
            blocks.extend(self._consume_resampled_frame(frame))
        if self._reservoir.size:
            final = np.zeros(self.block_samples, dtype=np.int16)
            final[: self._reservoir.size] = self._reservoir
            blocks.append(final)
        self._reservoir = np.empty(0, dtype=np.int16)
        self.close()
        return blocks

    def close(self) -> None:
        self._finished = True
        self._reservoir = np.empty(0, dtype=np.int16)
        self._codec = None
        self._resampler = None

    def _decode_packets(self, packets: list[Any]) -> list[np.ndarray]:
        if self._codec is None:
            raise RuntimeError("decoder is closed")
        blocks: list[np.ndarray] = []
        for packet in packets:
            if bytes(packet).startswith(b"ID3"):
                continue
            for frame in self._codec.decode(packet):
                blocks.extend(self._consume_decoded_frame(frame))
        return blocks

    def _consume_decoded_frame(self, frame: Any) -> list[np.ndarray]:
        if self._resampler is None:
            raise RuntimeError("decoder is closed")
        blocks: list[np.ndarray] = []
        for resampled in self._resampler.resample(frame):
            blocks.extend(self._consume_resampled_frame(resampled))
        return blocks

    def _consume_resampled_frame(self, frame: Any) -> list[np.ndarray]:
        samples = np.asarray(frame.to_ndarray(), dtype=np.int16).reshape(-1)
        blocks: list[np.ndarray] = []
        offset = 0
        while offset < samples.size:
            space = self._reservoir_limit - self._reservoir.size
            if space <= 0:
                raise RuntimeError("decoded PCM reservoir exceeded its bound")
            take = min(space, samples.size - offset)
            self._reservoir = np.concatenate((self._reservoir, samples[offset : offset + take]))
            offset += take
            while self._reservoir.size >= self.block_samples:
                blocks.append(self._reservoir[: self.block_samples].copy())
                self._reservoir = self._reservoir[self.block_samples :].copy()
        return blocks
