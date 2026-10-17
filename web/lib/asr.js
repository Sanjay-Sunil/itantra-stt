// Language ID (AmberNet) and IndicConformer ASR, ported from src/indicstt/lid.py and asr.py

import { logMel, DEFAULT_MEL, INDIC_CONFORMER_MEL } from "./features.js";

export class AmberNetLID {
  static async load(rt) {
    const m = new AmberNetLID();
    m.rt = rt;
    m.sess = await rt.session("lid-ambernet/ambernet.int8.onnx", { arena: false });
    m.labels = await rt.json("lid-ambernet/labels.json");
    return m;
  }

  /** Softmax over the languages this model knows → { scores, ms } */
  async scores(audio) {
    const t = performance.now(), rt = this.rt;
    const { data, frames } = logMel(audio, DEFAULT_MEL);
    const out = await rt.run("lid", this.sess, { features: rt.f32(data, [1, 80, frames]), length: rt.i64([frames], [1]) });
    rt.count("lid", "lid", frames);
    const lg = out.logits.data;
    let mx = -Infinity;
    for (const v of lg) mx = Math.max(mx, v);
    const e = Array.from(lg, (v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    const scores = {};
    this.labels.forEach((l, i) => (scores[l] = e[i] / s));
    return { scores, ms: performance.now() - t };
  }
}

export const BLANK = 256;
const SOS = 5632;
const MAX_SYMBOLS = 10;
export const INDIC_ASR = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn"];
const D = 1024, H = 640;

export class IndicConformer {
  static async load(rt, { tier = "int8", languages = [] } = {}) {
    const m = new IndicConformer();
    const a = `indic-conformer-${tier}/assets/`;
    m.rt = rt;
    m.a = a;
    m.tier = tier;
    m.encoder = await rt.session(a + "encoder.onnx", { arena: false });
    m.ctc = await rt.session(a + "ctc_decoder.onnx", { arena: false });
    m.pred = await rt.session(a + "rnnt_decoder.onnx");
    m.jointEnc = await rt.session(a + "joint_enc.onnx", { arena: false });
    m.jointPred = await rt.session(a + "joint_pred.onnx");
    m.jointPre = await rt.session(a + "joint_pre_net.onnx");
    m.vocab = await rt.json(a + "vocab.json");
    const masks = await rt.json(a + "language_masks.json");
    m.maskIdx = {};
    for (const [k, v] of Object.entries(masks)) m.maskIdx[k] = v.flatMap((b, i) => (b ? [i] : []));
    m.post = {};
    for (const l of languages) if (INDIC_ASR.includes(l)) await m.postNet(l);
    return m;
  }

  async postNet(lang) {
    if (!this.post[lang]) this.post[lang] = await this.rt.session(`${this.a}joint_post_net_${lang}.onnx`);
    return this.post[lang];
  }

  async encode(audio) {
    const rt = this.rt;
    const { data, frames } = logMel(audio, INDIC_CONFORMER_MEL);
    const o = await rt.run("encoder", this.encoder,
      { audio_signal: rt.f32(data, [1, 80, frames]), length: rt.i64([frames], [1]) });
    rt.count("encoder", "encoder", frames);
    const T = o.outputs.dims[2];
    return { out: o.outputs.data, T, length: Number(o.encoded_lengths.data[0]), ctc: null };
  }

  async ctcLogprobs(enc) {
    if (!enc.ctc) {
      const o = await this.rt.run("ctc", this.ctc, { encoder_output: this.rt.f32(enc.out, [1, D, enc.T]) });
      this.rt.count("ctc", "ctc", enc.T);
      enc.V = o.logprobs.dims[2];
      enc.ctc = o.logprobs.data;
    }
    return enc.ctc;
  }

  // per-frame log-softmax over one language's sub-vocabulary → { argmax, best } per frame
  _sub(enc, lang) {
    const idx = this.maskIdx[lang], V = enc.V, lp = enc.ctc;
    const res = [];
    for (let t = 0; t < enc.length; t++) {
      const row = t * V;
      let mx = -Infinity, am = 0;
      for (let j = 0; j < idx.length; j++) {
        const v = lp[row + idx[j]];
        if (v > mx) { mx = v; am = j; }
      }
      let s = 0;
      for (let j = 0; j < idx.length; j++) s += Math.exp(lp[row + idx[j]] - mx);
      res.push({ argmax: am, best: -Math.log(s) }); // max of log-softmax = mx - mx - log(sum)
    }
    return res;
  }

  /** Mean greedy log-prob of non-blank frames under each language's vocabulary (higher = better fit). */
  async ctcLanguageScores(enc, langs) {
    await this.ctcLogprobs(enc);
    const out = {};
    for (const l of langs) {
      if (!this.maskIdx[l]) continue;
      const sub = this._sub(enc, l);
      const nb = sub.filter((f) => f.argmax !== BLANK);
      const use = nb.length ? nb : sub;
      out[l] = use.reduce((a, f) => a + f.best, 0) / use.length;
    }
    return out;
  }

  detok(ids, lang) {
    const v = this.vocab[lang];
    return ids.map((i) => v[i]).join("").replaceAll("▁", " ").trim();
  }

  async decode(enc, lang, mode = "rnnt") {
    return mode === "rnnt" ? this._rnnt(enc, lang) : this._ctcDecode(enc, lang);
  }

  async _ctcDecode(enc, lang) {
    await this.ctcLogprobs(enc);
    const path = this._sub(enc, lang).map((f) => f.argmax);
    const ids = path.filter((p, i) => (i === 0 || p !== path[i - 1]) && p !== BLANK);
    return this.detok(ids, lang);
  }

  async _rnnt(enc, lang) {
    const rt = this.rt, post = await this.postNet(lang);
    const L = enc.length, T = enc.T;
    // (1, 1024, T) → (1, L, 1024)
    const x = new Float32Array(L * D);
    for (let t = 0; t < L; t++) for (let d = 0; d < D; d++) x[t * D + d] = enc.out[d * T + t];
    const fAll = (await rt.run("rnnt_joint", this.jointEnc, { input: rt.f32(x, [1, L, D]) })).output;
    rt.count("rnnt_joint", "rnnt_joint_enc", L);
    const J = fAll.dims[2];
    let h = new Float32Array(2 * H), c = new Float32Array(2 * H);
    let last = SOS;
    const out = [];
    let gCache = null;
    const predict = async (token, h, c) => {
      const o = await rt.run("rnnt_pred", this.pred, {
        targets: rt.i32([token], [1, 1]), target_length: rt.i32([1], [1]),
        "states.1": rt.f32(h, [2, 1, H]), "onnx::Slice_3": rt.f32(c, [2, 1, H]) });
      const names = this.pred.outputNames; // outputs, prednet_lengths, states, <c>
      const g = o[names[0]]; // (1, 640, 1) → transpose is a no-op for a single step
      rt.count("rnnt_pred", "rnnt_pred");
      const gp = (await rt.run("rnnt_pred", this.jointPred, { input: rt.f32(g.data, [1, 1, g.dims[1]]) })).output.data;
      return [gp, Float32Array.from(o[names[2]].data), Float32Array.from(o[names[3]].data)];
    };
    const z = new Float32Array(J);
    for (let t = 0; t < L; t++) {
      for (let s = 0; s < MAX_SYMBOLS; s++) {
        if (!gCache) gCache = await predict(last, h, c);
        const [g, h2, c2] = gCache;
        for (let j = 0; j < J; j++) z[j] = fAll.data[t * J + j] + g[j];
        const zz = (await rt.run("rnnt_joint", this.jointPre, { input: rt.f32(z, [1, 1, J]) })).output;
        const logits = (await rt.run("rnnt_joint", post, { input: zz })).output.data;
        rt.count("rnnt_joint", "rnnt_joint");
        let tok = 0;
        for (let k = 1; k < logits.length; k++) if (logits[k] > logits[tok]) tok = k;
        if (tok === BLANK) break;
        last = tok; // per-language id, as upstream
        out.push(tok);
        h = h2; c = c2;
        gCache = null;
      }
    }
    return this.detok(out, lang);
  }
}
