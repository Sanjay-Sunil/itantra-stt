"""English ASR path: whisper-tiny.en ONNX INT8 (loaded only when English is selected)."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ortutil import plain_options

N_FFT, HOP, N_MELS, N_SAMPLES = 400, 160, 80, 480000
SOT, NO_TIMESTAMPS, EOT = 50257, 50362, 50256


@lru_cache(maxsize=1)
def _mel_filters() -> np.ndarray:
    """Slaney-scale, Slaney-normalised mel filterbank (same as librosa / WhisperFeatureExtractor)."""
    sr, fmin, fmax = 16000, 0.0, 8000.0

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
        return np.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        lin = m * (200.0 / 3)
        log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log, lin)

    fft_freqs = np.linspace(0, sr / 2, N_FFT // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), N_MELS + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_pts[2:] - mel_pts[:-2]))[:, None]
    return weights.astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper log-mel features, padded/trimmed to 30 s → (1, 80, 3000)."""
    x = np.zeros(N_SAMPLES, dtype=np.float32)
    x[: min(len(audio), N_SAMPLES)] = audio[:N_SAMPLES]
    x = np.pad(x, N_FFT // 2, mode="reflect")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    n_frames = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n_frames, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = _mel_filters() @ power[:-1].T
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0)[None].astype(np.float32)


class WhisperEnglish:
    def __init__(self, path: str | Path, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        p = Path(path)
        so = plain_options(threads, prepack=True, arena=False)  # encoder: arena off −72 MB, same speed
        so_dec = plain_options(threads, prepack=True, arena=True)  # per-token loop: arena off is 3.5x slower
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(p / "onnx/encoder_model_int8.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(p / "onnx/decoder_model_merged_int8.onnx"), so_dec, providers=prov)
        self.tok = Tokenizer.from_file(str(p / "tokenizer.json"))
        self.past_names = [i.name for i in self.dec.get_inputs() if i.name.startswith("past_key_values")]
        self.n_layers = len(self.past_names) // 4

    def transcribe(self, audio: np.ndarray, max_tokens: int = 200) -> tuple[str, dict[str, float]]:
        t0 = time.perf_counter()
        hidden = self.enc.run(None, {"input_features": log_mel(audio)})[0]
        t1 = time.perf_counter()
        empty = np.zeros((1, 6, 0, 64), dtype=np.float32)
        past = {n: empty for n in self.past_names}
        ids = [SOT, NO_TIMESTAMPS]
        cur = np.array([ids], dtype=np.int64)
        out: list[int] = []
        for step in range(max_tokens):
            res = self.dec.run(None, {"input_ids": cur, "encoder_hidden_states": hidden,
                                      "use_cache_branch": np.array([step > 0]), **past})
            logits = res[0][0, -1]
            nxt = int(np.argmax(logits))
            if nxt == EOT:
                break
            out.append(nxt)
            new = dict(zip([o.name for o in self.dec.get_outputs()][1:], res[1:], strict=True))
            for name in self.past_names:
                key = name.replace("past_key_values", "present")
                # encoder KV come back empty on cached steps; keep the first ones
                if step == 0 or ".decoder." in name:
                    past[name] = new[key]
            cur = np.array([[nxt]], dtype=np.int64)
        text = self.tok.decode(out, skip_special_tokens=True).strip()
        return text, {"encoder": (t1 - t0) * 1000, "decode": (time.perf_counter() - t1) * 1000}
