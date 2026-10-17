// Per-segment pipeline: VAD → LID → IndicConformer encoder (or Whisper for English) → decode → MT.
// Same decisions as src/indicstt/engine.py (_Stages), run sequentially: LID goes first so an English segment
// skips the Indic encoder entirely (the Python engine runs both in parallel and cancels the encoder).

import { SileroVAD, GainControl, Segmenter } from "./vad.js";
import { AmberNetLID, IndicConformer, INDIC_ASR } from "./asr.js";
import { WhisperEnglish } from "./english.js";
import { Translator, bundleName } from "./mt.js";

export const LANGUAGES = { hi: "Hindi", gu: "Gujarati", mr: "Marathi", kn: "Kannada", ml: "Malayalam", ta: "Tamil",
  te: "Telugu", or: "Odia", bn: "Bengali", en: "English" };
export const INDIC = Object.keys(LANGUAGES).filter((l) => l !== "en");
const EN_SKIP_CONFIDENCE = 0.8;

/** Model files a configuration needs, so the page can show the download before loading. */
export function requiredModels(cfg) {
  const m = ["silero-vad"];
  if (!cfg.manualLanguage && cfg.languages.length > 1) m.push("lid-ambernet");
  if (cfg.languages.some((l) => INDIC.includes(l))) m.push(`indic-conformer-${cfg.asrTier || "int8"}`);
  if (cfg.languages.includes("en")) m.push("whisper-tiny-en");
  if (cfg.target) for (const s of cfg.languages) if (s !== cfg.target) m.push(bundleName(s, cfg.target));
  return [...new Set(m)];
}

export class Engine {
  /** cfg: { languages, target, manualLanguage, asrMode: "rnnt"|"ctc", lowConfidence, asrTier }.
   *  base: an engine whose loaded models are reused (only missing ones are loaded). */
  static async load(rt, cfg, onStage = () => {}, base = null) {
    const e = new Engine();
    e.rt = rt;
    e.cfg = { asrMode: "rnnt", lowConfidence: 0.5, asrTier: "int8", target: null, manualLanguage: null, ...cfg };
    const c = e.cfg;
    if (!base?.vad) onStage("vad");
    e.vad = base?.vad || (await SileroVAD.load(rt));
    if (!c.manualLanguage && c.languages.length > 1) {
      if (!base?.lid) onStage("lid");
      e.lid = base?.lid || (await AmberNetLID.load(rt));
    }
    if (c.languages.some((l) => INDIC.includes(l))) {
      const reuse = base?.indic?.tier === c.asrTier ? base.indic : null; // only the same speech-model tier
      if (!reuse) onStage("asr");
      e.indic = reuse || (await IndicConformer.load(rt, { tier: c.asrTier }));
      // per-language RNN-T heads: only the ones this configuration can use
      if (c.asrMode === "rnnt") for (const l of c.languages) if (INDIC_ASR.includes(l)) await e.indic.postNet(l);
      rt.releaseBuffers();
    }
    if (c.languages.includes("en")) {
      if (!base?.english) onStage("english");
      e.english = base?.english || (await WhisperEnglish.load(rt));
    }
    e.mt = base?.mt || new Translator(rt);
    if (c.target) for (const s of c.languages) if (s !== c.target) { onStage("mt"); await e.mt.bundle(s, c.target); }
    e.resetStream();
    return e;
  }

  resetStream() {
    this.vad.reset();
    const agc = new GainControl();
    this.vadMs = 0;
    this.vadTrace = []; // speech probability per 32 ms window, for the language-test view
    this.segmenter = new Segmenter(async (w) => {
      const t = performance.now();
      const p = await this.vad.score(agc.apply(w));
      this.vadMs += performance.now() - t;
      this.vadTrace.push(p);
      return p;
    });
  }

  /** Feed 16 kHz audio; returns finished speech segments (not yet transcribed). */
  async push(chunk) {
    return this._stamp(await this.segmenter.push(chunk));
  }

  flush() {
    return this._stamp(this.segmenter.flush());
  }

  _stamp(segs) {
    for (const s of segs) { s.timings.vad = this.vadMs; this.vadMs = 0; }
    return segs;
  }

  async _lidDecide(seg) {
    const c = this.cfg;
    if (c.manualLanguage) return { lang: c.manualLanguage, conf: 1, scores: null, decidedBy: "manual" };
    if (c.languages.length === 1) return { lang: c.languages[0], conf: 1, scores: null, decidedBy: "single" };
    const { scores, ms } = await this.lid.scores(seg.audio);
    seg.timings.lid = ms;
    const top = Object.keys(scores).reduce((a, k) => (scores[k] > scores[a] ? k : a));
    const allowed = Object.fromEntries(Object.entries(scores).filter(([k]) => c.languages.includes(k)));
    const indicSel = c.languages.filter((l) => INDIC.includes(l));
    const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
    const best = (o) => Object.keys(o).reduce((a, k) => (o[k] > o[a] ? k : a));
    const res = { scores, top, decidedBy: "lid" };
    if (indicSel.some((l) => !(l in scores))) {
      // a selected language AmberNet has no class for (Odia): decide English vs Indic here, CTC picks the Indic one
      const pEn = c.languages.includes("en") ? scores.en || 0 : 0;
      const pInd = sum(Object.fromEntries(Object.entries(scores).filter(([k]) => INDIC.includes(k))));
      const total = pEn + pInd || 1;
      if (pEn > pInd) return { ...res, lang: "en", conf: pEn / total };
      if (indicSel.length === 1) return { ...res, lang: indicSel[0], conf: pInd / total };
      const known = Object.fromEntries(Object.entries(allowed).filter(([k]) => INDIC.includes(k)));
      if (!Object.keys(known).length) return { ...res, lang: indicSel[0], conf: 0 };
      const g = best(known);
      return { ...res, lang: g, conf: known[g] / (sum(known) || 1) };
    }
    if (Object.keys(allowed).length) { const l = best(allowed); return { ...res, lang: l, conf: allowed[l] / sum(allowed) }; }
    return { ...res, lang: c.languages[0], conf: 0 };
  }

  /** Transcribe (and translate) one segment in place. */
  async process(seg) {
    const c = this.cfg, tm = seg.timings, t0 = performance.now();
    let { lang, conf, scores, top, decidedBy } = await this._lidDecide(seg);
    seg.lidTop = scores ? Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
    if (top && !c.languages.includes(top)) seg.flags.push("unexpected_language");
    let enc = null;
    if (this.indic && !(lang === "en" && conf >= EN_SKIP_CONFIDENCE)) {
      let t = performance.now();
      enc = await this.indic.encode(seg.audio);
      tm.encoder = performance.now() - t;
      const indicSel = c.languages.filter((l) => INDIC.includes(l));
      const unsure = conf < 0.8 || indicSel.some((l) => !(l in (scores || {})));
      if (!c.manualLanguage && lang !== "en" && indicSel.length > 1 && unsure) {
        t = performance.now();
        const ctc = await this.indic.ctcLanguageScores(enc, indicSel);
        tm.lid_ctc = performance.now() - t;
        seg.ctcScores = ctc;
        const bestL = Object.keys(ctc).reduce((a, k) => (ctc[k] > ctc[a] ? k : a));
        const mx = ctc[bestL], ps = Object.values(ctc).map((v) => Math.exp(v - mx));
        const pBest = 1 / ps.reduce((a, b) => a + b, 0);
        if (bestL !== lang) { lang = bestL; conf = pBest; decidedBy = "ctc"; }
        else { conf = Math.max(conf, pBest); decidedBy = "lid+ctc"; }
      }
    } else if (this.indic) tm.encoder_skipped = 1;
    seg.language = lang;
    seg.confidence = conf;
    seg.decidedBy = decidedBy;
    if (conf < c.lowConfidence && !c.manualLanguage && c.languages.length > 1) seg.flags.push("low_confidence");
    const t = performance.now();
    if (lang === "en") {
      if (!this.english) seg.flags.push("asr_unavailable");
      else {
        const r = await this.english.transcribe(seg.audio);
        seg.text = r.text;
        tm.encoder = r.encoderMs;
        tm.decode = r.decodeMs;
      }
    } else if (enc) {
      seg.text = await this.indic.decode(enc, lang, c.asrMode);
      tm.decode = performance.now() - t;
    } else seg.flags.push("asr_unavailable");
    if (c.target && lang !== c.target && seg.text?.trim()) {
      const r = await this.mt.translate(seg.text, lang, c.target);
      seg.translation = r.text;
      seg.targetLanguage = c.target;
      tm.mt = r.ms;
    }
    tm.total = performance.now() - t0 + (tm.vad || 0);
    return seg;
  }
}
