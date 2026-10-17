// Compare the JS (onnxruntime-web) LID / IndicConformer / Whisper outputs against Python references.
//   node test/parity_asr.mjs <ref.json>   (ref produced by the Python engine on data/smoke clips)
import fs from "fs";
import path from "path";
import { nodeRuntime, readWav16k } from "./node_runtime.js";
import { AmberNetLID, IndicConformer, INDIC_ASR } from "../lib/asr.js";
import { WhisperEnglish } from "../lib/english.js";

const ROOT = path.resolve(import.meta.dirname, "../..");
const ref = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const rt = nodeRuntime(Number(process.env.THREADS || 8));
const lid = await AmberNetLID.load(rt);
const asr = await IndicConformer.load(rt, { languages: INDIC_ASR });
const en = await WhisperEnglish.load(rt);

// word error rate between two strings (edit distance over words / reference words)
function wer(a, b) {
  const x = a.split(/\s+/).filter(Boolean), y = b.split(/\s+/).filter(Boolean);
  const d = Array.from({ length: x.length + 1 }, (_, i) => [i, ...Array(y.length).fill(0)]);
  for (let j = 1; j <= y.length; j++) d[0][j] = j;
  for (let i = 1; i <= x.length; i++)
    for (let j = 1; j <= y.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1));
  return d[x.length][y.length] / Math.max(1, x.length);
}

let same = 0, total = 0, werSum = 0, lidSame = 0;
for (const r of ref) {
  const a = readWav16k(path.join(ROOT, r.file));
  const { scores } = await lid.scores(a);
  const top = Object.keys(scores).reduce((p, k) => (scores[k] > scores[p] ? k : p));
  lidSame += top === r.lid;
  const pairs = [];
  if (r.lang === "en") pairs.push(["en", (await en.transcribe(a)).text, r.en]);
  else {
    const e = await asr.encode(a);
    pairs.push(["ctc", await asr.decode(e, r.lang, "ctc"), r.ctc], ["rnnt", await asr.decode(e, r.lang, "rnnt"), r.rnnt]);
  }
  for (const [k, js, py] of pairs) {
    total++;
    same += js === py;
    const w = wer(py, js);
    werSum += w;
    if (js !== py) console.log(`DIFF ${r.file} ${k} wer=${(100 * w).toFixed(1)}%\n  py: ${py}\n  js: ${js}`);
  }
}
console.log(`identical ${same}/${total}, mean WER js-vs-py ${(100 * werSum / total).toFixed(2)}%, LID top-1 agree ${lidSame}/${ref.length}`);
const stats = [...rt.runStats].map(([k, v]) => `${k}: ${v.runs} runs ${(v.ms / 1000).toFixed(1)} s`);
console.log(stats.join(" | "));
