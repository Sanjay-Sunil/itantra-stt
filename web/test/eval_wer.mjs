// Word error rate of the browser engine (onnxruntime-web) on the smoke set, same normalisation as
// src/indicstt/evaluation.py, so it compares directly with results/quantization.md.
//   node test/eval_wer.mjs [rnnt|ctc] > ../results/browser-asr.json
import fs from "fs";
import path from "path";
import { nodeRuntime, readWav16k } from "./node_runtime.js";
import { IndicConformer, INDIC_ASR } from "../lib/asr.js";

const MODE = process.argv[2] || "rnnt";
const ROOT = path.resolve(import.meta.dirname, "../..");
const items = fs.readFileSync(path.join(ROOT, "data/smoke/manifest.jsonl"), "utf8").trim().split("\n").map(JSON.parse)
  .filter((x) => INDIC_ASR.includes(x.language));

const normalize = (t) => t.normalize("NFC").toLowerCase().replace(/[।॥]/g, " ")
  .replace(/[^\p{L}\p{M}\p{N}\s]/gu, " ").split(/\s+/).filter(Boolean);
function edits(r, h) {
  let d = Array.from({ length: h.length + 1 }, (_, j) => j);
  for (let i = 1; i <= r.length; i++) {
    let prev = d[0];
    d[0] = i;
    for (let j = 1; j <= h.length; j++) {
      const cur = Math.min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] !== h[j - 1]));
      prev = d[j];
      d[j] = cur;
    }
  }
  return d[h.length];
}

const rt = nodeRuntime(Number(process.env.THREADS || 8));
const asr = await IndicConformer.load(rt, { languages: INDIC_ASR });
const per = {};
let audioS = 0;
const t0 = performance.now();
for (const it of items) {
  const a = readWav16k(path.join(ROOT, it.audio));
  audioS += a.length / 16000;
  const hyp = await asr.decode(await asr.encode(a), it.language, MODE);
  const r = normalize(it.text), p = (per[it.language] ||= { errors: 0, words: 0, clips: 0 });
  p.errors += edits(r, normalize(hyp));
  p.words += r.length;
  p.clips += 1;
  process.stderr.write(`\r${it.language} ${p.clips}`);
}
const wer = Object.fromEntries(Object.entries(per).map(([k, v]) => [k, +(100 * v.errors / v.words).toFixed(2)]));
const mean = +(Object.values(wer).reduce((a, b) => a + b, 0) / Object.keys(wer).length).toFixed(2);
const wall = (performance.now() - t0) / 1000;
console.log(JSON.stringify({ runtime: "onnxruntime-web (wasm, Node)", mode: MODE, threads: Number(process.env.THREADS || 8),
  clips: items.length, audio_s: +audioS.toFixed(1), wer, mean_wer: mean, rtf: +(wall / audioS).toFixed(3) }, null, 1));
