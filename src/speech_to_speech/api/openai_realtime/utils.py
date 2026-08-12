import numpy as np
import soxr
from scipy.signal import resample_poly


def resample(audio_int16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM audio between sample rates using polyphase filtering."""
    if from_rate == to_rate:
        return audio_int16
    samples = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
    gcd = np.gcd(to_rate, from_rate)
    resampled = resample_poly(samples, up=to_rate // gcd, down=from_rate // gcd)
    return np.clip(resampled * 32768, -32768, 32767).astype(np.int16).tobytes()


class StreamingPCMResampler:
    """Stateful PCM16 resampling for one continuous audio response.

    ``resample_poly`` is suitable for complete buffers but restarts its filter
    for every call. Realtime output arrives in packets, so resetting that
    filter at each packet boundary introduces recurring audible seams. The
    streaming converter retains its filter state until ``finish`` flushes the
    response tail.
    """

    def __init__(self, from_rate: int, to_rate: int) -> None:
        if from_rate <= 0 or to_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._finished = False
        self._stream = None
        if from_rate != to_rate:
            self._stream = soxr.ResampleStream(
                from_rate,
                to_rate,
                num_channels=1,
                dtype="int16",
                quality="HQ",
            )

    def process(self, audio_int16: bytes) -> bytes:
        if self._finished:
            raise RuntimeError("resampler is already finished")
        if len(audio_int16) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM16 audio must contain complete samples")
        if self._stream is None:
            return audio_int16
        samples = np.frombuffer(audio_int16, dtype=np.int16)
        return self._stream.resample_chunk(samples, last=False).tobytes()

    def finish(self) -> bytes:
        if self._finished:
            return b""
        self._finished = True
        if self._stream is None:
            return b""
        empty = np.empty(0, dtype=np.int16)
        return self._stream.resample_chunk(empty, last=True).tobytes()
