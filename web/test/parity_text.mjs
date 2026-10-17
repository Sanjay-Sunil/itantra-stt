// IndicProcessor port vs Python: node test/parity_text.mjs <ip_ref.json>
import fs from "fs";
import { preprocess, postprocess } from "../lib/indicproc.js";
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let pre = 0, post = 0, shown = 0;
// placeholder numbers follow Python set iteration order, which is randomised per process
const ids = (s) => s.replace(/< ID\d+ >/g, "< ID >");
for (const c of cases) {
  const p = preprocess(c.text, c.src, c.tgt);
  const body = p.text.split(" ").slice(2).join(" ");
  const q = postprocess(body, c.src, p.map);
  pre += ids(p.text) === ids(c.pre);
  post += q === c.post;
  if ((ids(p.text) !== ids(c.pre) || q !== c.post) && shown++ < 8)
    console.log(`DIFF [${c.src}→${c.tgt}]\n text: ${c.text}\n pre py: ${c.pre}\n pre js: ${p.text}\n post py: ${c.post}\n post js: ${q}`);
}
console.log(`preprocess identical ${pre}/${cases.length}, postprocess identical ${post}/${cases.length}`);
