// Prepare web/ for serving: vendor the runtime files, write the model manifest, copy the demo clips.
//   node build.mjs [--models-url URL]   (models from ../models, or INDICSTT_MODELS_DIR; URL = where a hosted copy
//                                        fetches them, e.g. https://huggingface.co/<repo>/resolve/main/)
import fs from "fs";
import path from "path";

const WEB = import.meta.dirname;
const MODELS = process.env.INDICSTT_MODELS_DIR || path.resolve(WEB, "../models");
const nm = (p) => path.join(WEB, "node_modules", p);

// 1. vendor: onnxruntime-web (wasm build, threaded + SIMD), tokenizers.js, cross-origin-isolation service worker
const vendor = path.join(WEB, "vendor");
fs.mkdirSync(vendor, { recursive: true });
for (const f of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"])
  fs.copyFileSync(nm(`onnxruntime-web/dist/${f}`), path.join(vendor, f));
fs.copyFileSync(nm("@huggingface/tokenizers/dist/tokenizers.min.mjs"), path.join(vendor, "tokenizers.min.mjs"));
fs.copyFileSync(nm("coi-serviceworker/coi-serviceworker.min.js"), path.join(WEB, "coi-serviceworker.min.js"));

// 2. model manifest: the files the browser engine fetches, with sizes
const POST_NETS = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn"].map((l) => `assets/joint_post_net_${l}.onnx`);
const MT = ["encoder_model.onnx", "encoder_model.onnx.data", "decoder_model.onnx", "decoder_with_past_model.onnx",
  "decoder_shared.onnx.data", "tokenizer_src.json", "tokenizer_tgt.json", "tokenizer_config.json",
  "tokenizer_meta.json", "generation_config.json"];
// the two speech-model tiers have the same files: INT8 (accurate) and INT4-mixed (smaller)
const CONFORMER = ["assets/encoder.onnx", "assets/encoder.onnx.data", "assets/ctc_decoder.onnx", "assets/rnnt_decoder.onnx",
  "assets/joint_enc.onnx", "assets/joint_pred.onnx", "assets/joint_pre_net.onnx", "assets/vocab.json",
  "assets/language_masks.json", ...POST_NETS];
const FILES = {
  "silero-vad": ["onnx/model.onnx"],
  "lid-ambernet": ["ambernet.int8.onnx", "labels.json"],
  "indic-conformer-int8": CONFORMER,
  "indic-conformer-int4mixed": CONFORMER,
  "whisper-tiny-en": ["onnx/encoder_model_int8.onnx", "onnx/decoder_model_merged_int8.onnx", "tokenizer.json",
    "tokenizer_config.json"],
  "mt-indic-en": MT, "mt-en-indic": MT, "mt-indic-indic": MT,
};
const manifest = {};
for (const [dir, files] of Object.entries(FILES)) {
  manifest[dir] = {};
  for (const f of files) {
    const p = path.join(MODELS, dir, f);
    if (!fs.existsSync(p)) throw new Error(`missing ${p}: run scripts/fetch_models.py / quantize.py first`);
    manifest[dir][f] = fs.statSync(p).size;
  }
}
fs.writeFileSync(path.join(WEB, "models.json"), JSON.stringify(manifest, null, 1));

// 3. sample clips (FLEURS, CC-BY-4.0): two short read sentences per language, from which the page assembles a
//    sample for whatever languages are selected; plus the fixed demo clips if present
const samples = path.join(WEB, "samples");
fs.rmSync(path.join(samples, "clips"), { recursive: true, force: true });
fs.mkdirSync(path.join(samples, "clips"), { recursive: true });
const smoke = path.resolve(WEB, "../data/smoke/manifest.jsonl");
const clips = {};
if (fs.existsSync(smoke)) {
  const wavSeconds = (f) => (fs.statSync(f).size - 44) / 32000; // 16 kHz mono PCM16
  const items = fs.readFileSync(smoke, "utf8").trim().split("\n").map(JSON.parse);
  for (const lang of ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"]) {
    const pick = items.filter((x) => x.language === lang)
      .map((x) => ({ ...x, seconds: wavSeconds(path.resolve(WEB, "..", x.audio)) }))
      .filter((x) => x.seconds >= 3.5 && x.seconds <= 10)
      .sort((a, b) => Math.abs(a.seconds - 6) - Math.abs(b.seconds - 6)).slice(0, 2);
    clips[lang] = pick.map((x, i) => {
      const file = `clips/${lang}_${i}.wav`;
      fs.copyFileSync(path.resolve(WEB, "..", x.audio), path.join(samples, file));
      return { file, text: x.text, seconds: +x.seconds.toFixed(2) };
    });
  }
}
fs.writeFileSync(path.join(samples, "clips.json"), JSON.stringify(clips, null, 1));
for (const f of ["mixed_en_hi_ta.wav", "all10_languages.wav"]) {
  const src = path.resolve(WEB, "../data/demo", f);
  if (fs.existsSync(src)) fs.copyFileSync(src, path.join(samples, f));
}
// 4. hosted copies: where to fetch the models from
const i = process.argv.indexOf("--models-url");
if (i > 0) fs.writeFileSync(path.join(WEB, "site.json"), JSON.stringify({ modelBase: process.argv[i + 1] }));
else fs.rmSync(path.join(WEB, "site.json"), { force: true });

const total = Object.values(manifest).flatMap(Object.values).reduce((a, b) => a + b, 0);
console.log(`vendor/ ready, models.json: ${Object.keys(manifest).length} models, ${(total / 1e6).toFixed(0)} MB total`);
