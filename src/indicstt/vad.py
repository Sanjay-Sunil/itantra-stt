"""Silero VAD + debounced segmenter: continuous 16 kHz stream → numbered speech segments."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .segment import FLAG_FORCED_CUT, Segment

WINDOW = 512  # 32 ms at 16 kHz
WINDOW_MS = WINDOW * 1000 / SAMPLE_RATE


class SileroVAD:
    """Silero VAD v5 ONNX: 512-sample windows, recurrent state kept across calls."""

    CONTEXT = 64

    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self.context, window.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(
            None, {"input": x, "state": self.state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self.context = x[:, -self.CONTEXT :]
        return float(out.reshape(-1)[0])


class GainControl:
    """Automatic gain on the VAD's input only (ASR still gets the raw audio).

    Tracks a peak envelope with ~1 s release and scales windows toward `target`, so a quiet speaker right after
    a loud one is still detected. Measured: no false segments on 20 s of hiss or low rumble (results/vad.md).
    """

    def __init__(self, target: float = 0.3, max_gain: float = 30.0, floor: float = 1e-3, release: float = 0.97):
        self.target, self.max_gain, self.floor, self.release = target, max_gain, floor, release
        self.env = floor

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.env = max(float(np.abs(window).max()), self.env * self.release, self.floor)
        return window * min(self.max_gain, self.target / self.env)


@dataclass
class VADConfig:
    threshold: float = 0.5  # start speech when p >= threshold
    neg_threshold: float = 0.35  # count silence when p < neg_threshold
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    speech_pad_ms: int = 96
    max_segment_s: float = 30.0
    agc: bool = True  # GainControl before Silero


class Segmenter:
    """Debounced state machine over per-window speech probabilities.

    Accepts pushed chunks of any size; output does not depend on chunk size.
    """

    def __init__(self, scorer: Callable[[np.ndarray], float], config: VADConfig | None = None):
        self.scorer = scorer
        self.cfg = config or VADConfig()
        c = self.cfg
        self.pad_win = max(0, round(c.speech_pad_ms / WINDOW_MS))
        self.min_sil_win = max(1, round(c.min_silence_ms / WINDOW_MS))
        self.min_speech_win = max(1, round(c.min_speech_ms / WINDOW_MS))
        self.max_win = max(1, int(c.max_segment_s * 1000 / WINDOW_MS))
        self.seq = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._win_idx = 0  # index of next window in the stream
        self._pre: deque[np.ndarray] = deque(maxlen=self.pad_win)
        self._buf: list[np.ndarray] = []
        self._start_win = 0
        self._silence = 0
        self._speech_wins = 0
        self._buf_tail: list[np.ndarray] = []
        self.triggered = False
        self.last_prob = 0.0

    def push(self, chunk: np.ndarray) -> Iterator[Segment]:
        self._pending = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32)])
        n = len(self._pending) // WINDOW
        for i in range(n):
            yield from self._step(self._pending[i * WINDOW : (i + 1) * WINDOW])
        self._pending = self._pending[n * WINDOW :]

    def flush(self) -> Iterator[Segment]:
        if self.triggered:
            seg = self._finish(trim=self._silence, forced=False)
            if seg:
                yield seg
        self.triggered = False

    def _step(self, win: np.ndarray) -> Iterator[Segment]:
        p = self.scorer(win)
        self.last_prob = p
        idx = self._win_idx
        self._win_idx += 1
        if not self.triggered:
            if p >= self.cfg.threshold:
                self.triggered = True
                self._buf = [*self._pre, win]
                self._start_win = idx - len(self._pre)
                self._silence = 0
                self._speech_wins = 1
                self._pre.clear()
            else:
                self._pre.append(win)
            return
        self._buf.append(win)
        if p < self.cfg.neg_threshold:
            self._silence += 1
        else:
            if p >= self.cfg.threshold:
                self._speech_wins += 1
            self._silence = 0
        if self._silence >= self.min_sil_win:
            seg = self._finish(trim=self._silence, forced=False)
            self.triggered = False
            # keep the trailing silence as pre-roll for the next segment
            self._pre.extend(self._buf_tail)
            if seg:
                yield seg
        elif len(self._buf) >= self.max_win:
            seg = self._finish(trim=0, forced=True)
            # continue speech immediately in a new segment
            self._buf, self._start_win, self._silence, self._speech_wins = [], self._win_idx, 0, 0
            if seg:
                yield seg

    def _finish(self, trim: int, forced: bool) -> Segment | None:
        keep = len(self._buf) - max(0, trim - self.pad_win)
        self._buf_tail = self._buf[keep:][-self.pad_win :] if self.pad_win else []
        wins = self._buf[:keep]
        self._buf = []
        if forced:
            self.triggered = True
        if self._speech_wins < self.min_speech_win:
            return None
        audio = np.concatenate(wins)
        start_ms = int(self._start_win * WINDOW_MS)
        seg = Segment(seq=self.seq, start_ms=start_ms, end_ms=start_ms + int(len(wins) * WINDOW_MS), audio=audio)
        if forced:
            seg.flags.append(FLAG_FORCED_CUT)
        self.seq += 1
        return seg
