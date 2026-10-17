// Environment glue shared by Node (tests) and the browser worker.
// rt = { ort, Tokenizer, fetchBytes(path) → Uint8Array, onSession?(info) }

// Model files whose weights live in a separate external-data file
const EXTERNAL = {
  "indic-conformer-int8/assets/encoder.onnx": "encoder.onnx.data",
  "indic-conformer-int4mixed/assets/encoder.onnx": "encoder.onnx.data",
};
function externalFor(path) {
  if (EXTERNAL[path]) return EXTERNAL[path];
  if (/^mt-[a-z-]+\/encoder_model\.onnx$/.test(path)) return "encoder_model.onnx.data";
  if (/^mt-[a-z-]+\/decoder(_with_past)?_model\.onnx$/.test(path)) return "decoder_shared.onnx.data";
  return null;
}

const decoder = new TextDecoder();

export class Runtime {
  constructor({ ort, Tokenizer, fetchBytes }) {
    Object.assign(this, { ort, Tokenizer, fetchBytes });
    this.sessions = []; // { path, bytes, loadMs }
    this.runStats = new Map(); // label → { runs, ms, macs }
    this.compute = null; // web/compute.json: MAC polynomials per model, from scripts/count_macs.py
    this._ext = new Map(); // shared external-data buffers (MT decoders share one file)
  }

  async json(path) {
    return JSON.parse(decoder.decode(await this.fetchBytes(path)));
  }

  async session(path, { arena = true } = {}) {
    const t = performance.now();
    const model = await this.fetchBytes(path);
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all", enableCpuMemArena: arena };
    const ext = externalFor(path);
    let bytes = model.length;
    if (ext) {
      const dir = path.slice(0, path.lastIndexOf("/") + 1);
      const key = dir + ext;
      if (!this._ext.has(key)) this._ext.set(key, this.fetchBytes(key));
      const data = await this._ext.get(key);
      bytes += data.length;
      opts.externalData = [{ path: ext, data }];
    }
    const sess = await this.ort.InferenceSession.create(model, opts);
    this.sessions.push({ path, bytes, loadMs: performance.now() - t });
    return sess;
  }

  /** Drop references to downloaded external-data buffers once every session that needs them exists. */
  releaseBuffers() {
    this._ext.clear();
  }

  async run(label, sess, feeds, fetches) {
    const t = performance.now();
    const out = await sess.run(feeds, fetches);
    const s = this._stat(label);
    s.runs += 1;
    s.ms += performance.now() - t;
    this.runStats.set(label, s);
    return out;
  }

  _stat(label) {
    if (!this.runStats.has(label)) this.runStats.set(label, { runs: 0, ms: 0, macs: 0 });
    return this.runStats.get(label);
  }

  /** Add the multiply-accumulates of one run of model `key` at input size x (see scripts/count_macs.py). */
  count(label, key, x = 0) {
    const c = this.compute?.[key]?.coef;
    if (c) this._stat(label).macs += c[0] + c[1] * x + c[2] * x * x;
  }

  f32(data, dims) { return new this.ort.Tensor("float32", data, dims); }
  i64(values, dims) { return new this.ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims); }
  i32(values, dims) { return new this.ort.Tensor("int32", Int32Array.from(values), dims); }
}
