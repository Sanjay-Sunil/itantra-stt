// IndicTrans2 port vs Python: node test/parity_mt.mjs <mt_ref.json>
import fs from "fs";
import { nodeRuntime } from "./node_runtime.js";
import { Translator } from "../lib/mt.js";
import { preprocess } from "../lib/indicproc.js";

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const rt = nodeRuntime(Number(process.env.THREADS || 8));
const tr = new Translator(rt);
let ids = 0, same = 0, ms = 0;
for (const c of cases) {
  const b = await tr.bundle(c.src, c.tgt);
  const js = b.srcTok.encode(preprocess(c.text, c.src, c.tgt).text).ids;
  const idOk = JSON.stringify(js) === JSON.stringify(c.ids);
  ids += idOk;
  const r = await tr.translate(c.text, c.src, c.tgt);
  ms += r.ms;
  same += r.text === c.mt;
  if (!idOk) console.log(`IDS DIFF [${c.src}→${c.tgt}] py ${c.ids.length} js ${js.length}`);
  if (r.text !== c.mt) console.log(`DIFF [${c.src}→${c.tgt}]\n  py: ${c.mt}\n  js: ${r.text}`);
}
console.log(`token ids identical ${ids}/${cases.length}, translations identical ${same}/${cases.length}, mean ${(ms / cases.length) | 0} ms`);
