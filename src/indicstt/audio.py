"""Audio input: files, buffers and the microphone, delivered as 16 kHz mono float32 chunks."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def load_audio(path_or_file) -> np.ndarray:
    """Decode any soundfile-supported file (wav, flac, ogg, mp3) to 16 kHz mono float32."""
    audio, sr = sf.read(path_or_file, dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr)


class AudioSource:
    """Yields float32 chunks at 16 kHz until exhausted or stopped."""

    def chunks(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class BufferSource(AudioSource):
    def __init__(self, audio: np.ndarray, chunk_ms: int = 100):
        self.audio = audio
        self.chunk = SAMPLE_RATE * chunk_ms // 1000

    def chunks(self) -> Iterator[np.ndarray]:
        for i in range(0, len(self.audio), self.chunk):
            yield self.audio[i : i + self.chunk]


class FileSource(BufferSource):
    def __init__(self, path, chunk_ms: int = 100):
        super().__init__(load_audio(path), chunk_ms)


class MicSource(AudioSource):
    """Live microphone capture via sounddevice (optional dependency)."""

    def __init__(self, chunk_ms: int = 32, device=None):
        self.blocksize = SAMPLE_RATE * chunk_ms // 1000
        self.device = device
        self._q: queue.Queue = queue.Queue()
        self._running = False

    def chunks(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        def cb(indata, frames, t, status):
            self._q.put(indata[:, 0].copy())

        self._running = True
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=self.blocksize, device=self.device, callback=cb,
        ):
            while self._running:
                try:
                    yield self._q.get(timeout=0.2)
                except queue.Empty:
                    continue

    def stop(self) -> None:
        self._running = False
