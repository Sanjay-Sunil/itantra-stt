// Log-mel features, ported from src/indicstt/features.py (NeMo FilterbankFeatures) and english.py (Whisper).
// Arithmetic is float64 like the numpy reference; outputs are Float32Array in row-major (n_mels, frames).

function slaneyMelFb(sr, nFft, nMels, fmin = 0, fmax = sr / 2) {
  const hzToMel = (f) => (f >= 1000 ? 15 + Math.log(Math.max(f, 1e-10) / 1000) / (Math.log(6.4) / 27) : f / (200 / 3));
  const melToHz = (m) => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : m * (200 / 3));
  const nBins = nFft / 2 + 1;
  const mMin = hzToMel(fmin), mMax = hzToMel(fmax);
  const pts = [];
  for (let i = 0; i < nMels + 2; i++) pts.push(melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1)));
  const fb = new Float64Array(nMels * nBins);
  for (let m = 0; m < nMels; m++) {
    const enorm = 2 / (pts[m + 2] - pts[m]);
    for (let k = 0; k < nBins; k++) {
      const f = ((sr / 2) * k) / (nBins - 1);
      const lower = (f - pts[m]) / (pts[m + 1] - pts[m]);
      const upper = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
      fb[m * nBins + k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
  }
  // numpy builds the bank in float64 then casts to float32 before use
  return Float64Array.from(Float32Array.from(fb));
}

// In-place iterative radix-2 complex FFT
function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const a = i + k, b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

export const DEFAULT_MEL = { sampleRate: 16000, nFft: 512, winLength: 400, hop: 160, nMels: 80, preemph: 0.97,
  logGuard: 2 ** -24, normalize: true, padMode: "constant", extraFrame: false };
export const INDIC_CONFORMER_MEL = { ...DEFAULT_MEL, padMode: "reflect", extraFrame: true };

const cache = new Map();
function nemoTables(cfg) {
  const key = `${cfg.sampleRate}/${cfg.nFft}/${cfg.nMels}/${cfg.winLength}`;
  if (!cache.has(key)) {
    const fb = slaneyMelFb(cfg.sampleRate, cfg.nFft, cfg.nMels);
    const win = new Float64Array(cfg.nFft);
    const left = (cfg.nFft - cfg.winLength) >> 1;
    for (let n = 0; n < cfg.winLength; n++) win[left + n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (cfg.winLength - 1));
    cache.set(key, { fb, win });
  }
  return cache.get(key);
}

function pad(x, p, mode) {
  const out = new Float64Array(x.length + 2 * p);
  out.set(x, p);
  if (mode === "reflect") {
    for (let i = 0; i < p; i++) {
      out[p - 1 - i] = x[Math.min(i + 1, x.length - 1)];
      out[p + x.length + i] = x[Math.max(x.length - 2 - i, 0)];
    }
  }
  return out;
}

/** NeMo log-mel: audio Float32Array → { data: Float32Array(nMels*frames), frames } */
export function logMel(audio, cfg = DEFAULT_MEL) {
  const n = audio.length;
  let x = Float64Array.from(audio);
  if (cfg.preemph) {
    const y = new Float64Array(n);
    y[0] = x[0];
    for (let i = 1; i < n; i++) y[i] = x[i] - cfg.preemph * x[i - 1];
    x = y;
  }
  const { fb, win } = nemoTables(cfg);
  const nFft = cfg.nFft, nBins = nFft / 2 + 1, M = cfg.nMels;
  x = pad(x, nFft / 2, cfg.padMode);
  const nFrames = 1 + Math.floor((x.length - nFft) / cfg.hop);
  const seqLen = Math.max(1, Math.floor(n / cfg.hop) + (cfg.extraFrame ? 1 : 0));
  const F = Math.min(seqLen, nFrames);
  const mel = new Float64Array(M * seqLen);
  const re = new Float64Array(nFft), im = new Float64Array(nFft), pw = new Float64Array(nBins);
  for (let t = 0; t < F; t++) {
    const off = t * cfg.hop;
    for (let i = 0; i < nFft; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
    fftRadix2(re, im);
    for (let k = 0; k < nBins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < M; m++) {
      let s = 0;
      const row = m * nBins;
      for (let k = 0; k < nBins; k++) s += fb[row + k] * pw[k];
      mel[m * seqLen + t] = Math.log(s + cfg.logGuard);
    }
  }
  const out = new Float32Array(M * seqLen);
  for (let m = 0; m < M; m++) {
    const row = m * seqLen;
    if (!cfg.normalize) {
      for (let t = 0; t < seqLen; t++) out[row + t] = mel[row + t];
      continue;
    }
    let mean = 0;
    for (let t = 0; t < seqLen; t++) mean += mel[row + t];
    mean /= seqLen;
    let v = 0;
    for (let t = 0; t < seqLen; t++) v += (mel[row + t] - mean) ** 2;
    const std = Math.sqrt(Math.max(v / Math.max(1, seqLen - 1), 2 ** -24)) + 1e-5;
    for (let t = 0; t < seqLen; t++) out[row + t] = (mel[row + t] - mean) / std;
  }
  return { data: out, frames: seqLen };
}

// ---------------------------------------------------------------- Whisper (n_fft 400: direct real DFT tables)
const W = { nFft: 400, hop: 160, nMels: 80, nSamples: 480000 };
let whisperTables = null;
function getWhisperTables() {
  if (!whisperTables) {
    const nBins = W.nFft / 2 + 1;
    const cos = new Float64Array(nBins * W.nFft), sin = new Float64Array(nBins * W.nFft);
    for (let k = 0; k < nBins; k++)
      for (let i = 0; i < W.nFft; i++) {
        const a = (-2 * Math.PI * k * i) / W.nFft;
        cos[k * W.nFft + i] = Math.cos(a);
        sin[k * W.nFft + i] = Math.sin(a);
      }
    const win = new Float64Array(W.nFft);
    for (let i = 0; i < W.nFft; i++) win[i] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / W.nFft));
    whisperTables = { cos, sin, win, fb: slaneyMelFb(16000, W.nFft, W.nMels, 0, 8000) };
  }
  return whisperTables;
}

/** Whisper log-mel padded/trimmed to 30 s → Float32Array(80*3000) */
export function whisperLogMel(audio) {
  const { cos, sin, win, fb } = getWhisperTables();
  const nBins = W.nFft / 2 + 1, T = 3000;
  const x0 = new Float32Array(W.nSamples);
  x0.set(audio.subarray(0, Math.min(audio.length, W.nSamples)));
  const x = pad(x0, W.nFft / 2, "reflect");
  // frames entirely inside the zero padding have zero power: skip their DFT
  const lastLive = Math.min(T, Math.ceil((Math.min(audio.length, W.nSamples) + W.nFft) / W.hop) + 1);
  const logm = new Float64Array(W.nMels * T).fill(-10);
  const pw = new Float64Array(nBins), fr = new Float64Array(W.nFft);
  let mx = -Infinity;
  for (let t = 0; t < lastLive; t++) {
    const off = t * W.hop;
    for (let i = 0; i < W.nFft; i++) fr[i] = Math.fround(x[off + i] * win[i]);
    for (let k = 0; k < nBins; k++) {
      let r = 0, im = 0;
      const row = k * W.nFft;
      for (let i = 0; i < W.nFft; i++) { r += fr[i] * cos[row + i]; im += fr[i] * sin[row + i]; }
      pw[k] = r * r + im * im;
    }
    for (let m = 0; m < W.nMels; m++) {
      let s = 0;
      const row = m * nBins;
      for (let k = 0; k < nBins; k++) s += fb[row + k] * pw[k];
      const v = Math.log10(Math.max(s, 1e-10));
      logm[m * T + t] = v;
    }
  }
  for (let i = 0; i < logm.length; i++) if (logm[i] > mx) mx = logm[i];
  const out = new Float32Array(W.nMels * T);
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(logm[i], mx - 8) + 4) / 4;
  return out;
}
