// Prepare web/ for serving: vendor the runtime files, write the model manifest, copy the demo clips.
//   node build.mjs            (models from ../models, or INDICSTT_MODELS_DIR)
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
const FILES = {
  "silero-vad": ["onnx/model.onnx"],
  "lid-ambernet": ["ambernet.int8.onnx", "labels.json"],
  "indic-conformer-int8": ["assets/encoder.onnx", "assets/encoder.onnx.data", "assets/ctc_decoder.onnx",
    "assets/rnnt_decoder.onnx", "assets/joint_enc.onnx", "assets/joint_pred.onnx", "assets/joint_pre_net.onnx",
    "assets/vocab.json", "assets/language_masks.json", ...POST_NETS],
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

// 3. demo clips (FLEURS, CC-BY-4.0), if present
const samples = path.join(WEB, "samples");
fs.mkdirSync(samples, { recursive: true });
for (const f of ["mixed_en_hi_ta.wav", "all10_languages.wav"]) {
  const src = path.resolve(WEB, "../data/demo", f);
  if (fs.existsSync(src)) fs.copyFileSync(src, path.join(samples, f));
}
const total = Object.values(manifest).flatMap(Object.values).reduce((a, b) => a + b, 0);
console.log(`vendor/ ready, models.json: ${Object.keys(manifest).length} models, ${(total / 1e6).toFixed(0)} MB total`);
