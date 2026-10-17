// Whole browser pipeline on a WAV file, compared with the Python engine's segments.
//   node test/pipeline.mjs <wav> <langs comma-separated> [ref.json]
import fs from "fs";
import path from "path";
import { nodeRuntime, readWav16k } from "./node_runtime.js";
import { Engine } from "../lib/engine.js";

const [wav, langs, refPath] = process.argv.slice(2);
const ROOT = path.resolve(import.meta.dirname, "../..");
const rt = nodeRuntime(Number(process.env.THREADS || 8));
let t = performance.now();
const eng = await Engine.load(rt, { languages: langs.split(","), target: "en" });
console.log(`load ${(performance.now() - t) | 0} ms`);
const audio = readWav16k(path.resolve(ROOT, wav));
t = performance.now();
const segs = [];
for (let i = 0; i < audio.length; i += 4096) segs.push(...(await eng.push(audio.subarray(i, i + 4096))));
segs.push(...eng.flush());
for (const s of segs) await eng.process(s);
const wall = performance.now() - t;
const ref = refPath ? JSON.parse(fs.readFileSync(refPath, "utf8"))[wav] : null;
let sameLang = 0, sameText = 0;
for (const s of segs) {
  const r = ref?.find((x) => x.seq === s.seq && x.start === s.startMs);
  sameLang += r?.lang === s.language;
  sameText += r?.text === s.text;
  console.log(`${s.seq} ${s.startMs}-${s.endMs} ${s.language} ${s.decidedBy}${r && r.lang !== s.language ? ` (py ${r.lang})` : ""} ` +
    `${s.text.slice(0, 60)} | ${(s.translation || "").slice(0, 50)}`);
}
if (ref) console.log(`segments js ${segs.length} / py ${ref.length}; language agree ${sameLang}; text identical ${sameText}`);
console.log(`audio ${(audio.length / 16000).toFixed(1)} s, processing ${(wall / 1000).toFixed(1)} s, RTF ${(wall / 1000 / (audio.length / 16000)).toFixed(2)}`);
