"""NeMo-compatible log-mel features in numpy (FilterbankFeatures, eval mode, no dither).

Used by AmberNet LID; parity against NeMo is checked by scripts/export_ambernet.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def slaney_mel_fb(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0),
                        f / (200.0 / 3))

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * (200.0 / 3))

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(pts)
    ramps = pts[:, None] - fft_freqs[None, :]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    w *= (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return w.astype(np.float32)


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    preemph: float = 0.97
    log_guard: float = 2.0**-24
    normalize: str | None = "per_feature"


@lru_cache(maxsize=8)
def _fb_and_window(cfg: MelConfig):
    fb = slaney_mel_fb(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    n = np.arange(cfg.win_length)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / (cfg.win_length - 1))  # symmetric hann (periodic=False)
    left = (cfg.n_fft - cfg.win_length) // 2
    full = np.zeros(cfg.n_fft, dtype=np.float64)
    full[left : left + cfg.win_length] = win
    return fb, full


def log_mel(audio: np.ndarray, cfg: MelConfig = MelConfig()) -> np.ndarray:
    """audio (T,) float32 → features (n_mels, frames) float32."""
    x = np.asarray(audio, dtype=np.float64)
    if cfg.preemph:
        x = np.concatenate([x[:1], x[1:] - cfg.preemph * x[:-1]])
    fb, window = _fb_and_window(cfg)
    pad = cfg.n_fft // 2
    x = np.pad(x, pad, mode="constant")
    n_frames = 1 + (len(x) - cfg.n_fft) // cfg.hop_length
    frames = np.lib.stride_tricks.as_strided(
        x, (n_frames, cfg.n_fft), (x.strides[0] * cfg.hop_length, x.strides[0]), writeable=False
    )
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = np.log(fb.astype(np.float64) @ power.T + cfg.log_guard)
    seq_len = max(1, len(audio) // cfg.hop_length)  # NeMo get_seq_len: floor((T + 2*(n_fft//2) - n_fft) / hop)
    mel = mel[:, :seq_len]
    if cfg.normalize == "per_feature":
        mean = mel.mean(axis=1, keepdims=True)
        std = np.sqrt(((mel - mean) ** 2).sum(axis=1, keepdims=True) / max(1, seq_len - 1)) + 1e-5
        mel = (mel - mean) / std
    return mel.astype(np.float32)
