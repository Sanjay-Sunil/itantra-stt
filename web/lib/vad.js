// Silero VAD + debounced segmenter, ported from src/indicstt/vad.py

export const SAMPLE_RATE = 16000;
export const WINDOW = 512; // 32 ms
const WINDOW_MS = (WINDOW * 1000) / SAMPLE_RATE;
const CONTEXT = 64;

export class SileroVAD {
  static async load(rt) {
    const v = new SileroVAD();
    v.rt = rt;
    v.sess = await rt.session("silero-vad/onnx/model.onnx");
    v.reset();
    return v;
  }

  reset() {
    this.state = new Float32Array(2 * 128);
    this.context = new Float32Array(CONTEXT);
  }

  async score(win) {
    const x = new Float32Array(CONTEXT + WINDOW);
    x.set(this.context);
    x.set(win, CONTEXT);
    const rt = this.rt;
    const out = await rt.run("vad", this.sess, {
      input: rt.f32(x, [1, x.length]),
      state: rt.f32(this.state, [2, 1, 128]),
      sr: rt.i64([SAMPLE_RATE], []),
    });
    this.state = Float32Array.from(out.stateN.data);
    this.context = x.slice(x.length - CONTEXT);
    return out.output.data[0];
  }
}

/** Automatic gain on the VAD input only (ASR gets raw audio). */
export class GainControl {
  constructor(target = 0.3, maxGain = 10, floor = 1e-3, release = 0.97) {
    Object.assign(this, { target, maxGain, floor, release, env: floor });
  }
  apply(win) {
    let peak = 0;
    for (const v of win) peak = Math.max(peak, Math.abs(v));
    this.env = Math.max(peak, this.env * this.release, this.floor);
    const g = Math.min(this.maxGain, Math.max(1, this.target / this.env));
    return g === 1 ? win : win.map((v) => v * g);
  }
}

export const VAD_DEFAULTS = { threshold: 0.5, negThreshold: 0.35, minSpeechMs: 250, minSilenceMs: 400,
  speechPadMs: 96, maxSegmentS: 30, agc: true };

function concat(wins) {
  const out = new Float32Array(wins.length * WINDOW);
  wins.forEach((w, i) => out.set(w, i * WINDOW));
  return out;
}

/** Debounced state machine over per-window speech probabilities; output does not depend on chunk size. */
export class Segmenter {
  constructor(scorer, cfg = {}) {
    this.scorer = scorer; // async (win) => prob
    const c = (this.cfg = { ...VAD_DEFAULTS, ...cfg });
    this.padWin = Math.max(0, Math.round(c.speechPadMs / WINDOW_MS));
    this.minSilWin = Math.max(1, Math.round(c.minSilenceMs / WINDOW_MS));
    this.minSpeechWin = Math.max(1, Math.round(c.minSpeechMs / WINDOW_MS));
    this.maxWin = Math.max(1, Math.floor((c.maxSegmentS * 1000) / WINDOW_MS));
    this.seq = 0;
    this.pending = new Float32Array(0);
    this.winIdx = 0;
    this.pre = [];
    this.buf = [];
    this.bufTail = [];
    this.startWin = 0;
    this.silence = 0;
    this.speechWins = 0;
    this.triggered = false;
    this.lastProb = 0;
  }

  _pushPre(w) {
    this.pre.push(w);
    if (this.pre.length > this.padWin) this.pre.shift();
  }

  async push(chunk) {
    const p = new Float32Array(this.pending.length + chunk.length);
    p.set(this.pending);
    p.set(chunk, this.pending.length);
    const n = Math.floor(p.length / WINDOW);
    const segs = [];
    for (let i = 0; i < n; i++) {
      const s = await this._step(p.slice(i * WINDOW, (i + 1) * WINDOW));
      if (s) segs.push(s);
    }
    this.pending = p.slice(n * WINDOW);
    return segs;
  }

  flush() {
    const segs = [];
    if (this.triggered) {
      const s = this._finish(this.silence, false);
      if (s) segs.push(s);
    }
    this.triggered = false;
    return segs;
  }

  async _step(win) {
    const p = await this.scorer(win);
    this.lastProb = p;
    const idx = this.winIdx++;
    const c = this.cfg;
    if (!this.triggered) {
      if (p >= c.threshold) {
        this.triggered = true;
        this.buf = [...this.pre, win];
        this.startWin = idx - this.pre.length;
        this.silence = 0;
        this.speechWins = 1;
        this.pre = [];
      } else this._pushPre(win);
      return null;
    }
    this.buf.push(win);
    if (p < c.negThreshold) this.silence += 1;
    else {
      if (p >= c.threshold) this.speechWins += 1;
      this.silence = 0;
    }
    if (this.silence >= this.minSilWin) {
      const seg = this._finish(this.silence, false);
      this.triggered = false;
      for (const w of this.bufTail) this._pushPre(w); // trailing silence becomes the next segment's pre-roll
      return seg;
    }
    if (this.buf.length >= this.maxWin) {
      const seg = this._finish(0, true);
      this.buf = [];
      this.startWin = this.winIdx;
      this.silence = 0;
      this.speechWins = 0;
      return seg;
    }
    return null;
  }

  _finish(trim, forced) {
    const keep = this.buf.length - Math.max(0, trim - this.padWin);
    this.bufTail = this.padWin ? this.buf.slice(keep).slice(-this.padWin) : [];
    const wins = this.buf.slice(0, keep);
    this.buf = [];
    if (forced) this.triggered = true;
    if (this.speechWins < this.minSpeechWin) return null;
    const startMs = Math.floor(this.startWin * WINDOW_MS);
    const seg = { seq: this.seq++, startMs, endMs: startMs + Math.floor(wins.length * WINDOW_MS), audio: concat(wins),
      flags: forced ? ["forced_cut"] : [], timings: {} };
    return seg;
  }
}
