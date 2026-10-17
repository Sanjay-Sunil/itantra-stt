// IndicTrans2 distilled ONNX INT8 with greedy decoding and KV cache, ported from src/indicstt/mt.py

import { preprocess, postprocess } from "./indicproc.js";

export function bundleName(src, tgt) {
  if (src === "en") return "mt-en-indic";
  if (tgt === "en") return "mt-indic-en";
  return "mt-indic-indic";
}

export class IndicTrans {
  static async load(rt, name) {
    const m = new IndicTrans();
    const p = name + "/";
    m.rt = rt;
    m.name = name;
    m.enc = await rt.session(p + "encoder_model.onnx", { arena: false });
    m.dec = await rt.session(p + "decoder_model.onnx", { arena: false });
    m.decPast = await rt.session(p + "decoder_with_past_model.onnx", { arena: false });
    rt.releaseBuffers();
    m.nLayers = (m.dec.outputNames.length - 1) / 4;
    const cfg = await rt.json(p + "tokenizer_config.json");
    m.srcTok = new rt.Tokenizer(await rt.json(p + "tokenizer_src.json"), cfg);
    m.tgtTok = new rt.Tokenizer(await rt.json(p + "tokenizer_tgt.json"), cfg);
    m.meta = await rt.json(p + "tokenizer_meta.json");
    const gen = await rt.json(p + "generation_config.json");
    m.startId = gen.decoder_start_token_id ?? 2;
    m.eosId = gen.eos_token_id ?? 2;
    return m;
  }

  encodeIds(text) {
    const { unk_id: unk, src_dict_size: n } = this.meta;
    return this.srcTok.encode(text).ids.map((i) => (i < n ? i : unk));
  }

  async translate(text, src, tgt) {
    const rt = this.rt, label = "mt";
    const pre = preprocess(text, src, tgt);
    const ids = this.encodeIds(pre.text);
    const L = ids.length;
    const inputIds = rt.i64(ids, [1, L]), mask = rt.i64(Array(L).fill(1), [1, L]);
    const hidden = (await rt.run(label, this.enc, { input_ids: inputIds, attention_mask: mask })).last_hidden_state;
    rt.count(label, `${this.name}/enc`, L);
    const out = [this.startId];
    let past = null;
    const steps = Math.min(256, 2 * L + 16);
    for (let step = 0; step < steps; step++) {
      const cur = rt.i64([out[out.length - 1]], [1, 1]);
      let res;
      if (step === 0) res = await rt.run(label, this.dec, { input_ids: cur, encoder_hidden_states: hidden, encoder_attention_mask: mask });
      else {
        const feed = { input_ids: cur, encoder_attention_mask: mask };
        for (let i = 0; i < this.nLayers; i++)
          for (const kind of ["decoder.key", "decoder.value", "encoder.key", "encoder.value"])
            feed[`past_key_values.${i}.${kind}`] = past[`present.${i}.${kind}`];
        res = await rt.run(label, this.decPast, feed);
      }
      rt.count(label, `${this.name}/dec`, L);
      // decoder_with_past returns only the updated self-attention KV; encoder KV stay from step 0
      past = step === 0 ? res : { ...past, ...res };
      const lg = res.logits, V = lg.dims[2], off = (lg.dims[1] - 1) * V;
      let nxt = 0;
      for (let k = 1; k < V; k++) if (lg.data[off + k] > lg.data[off + nxt]) nxt = k;
      out.push(nxt);
      if (nxt === this.eosId) break;
    }
    const { unk_id: unk, tgt_dict_size: tn } = this.meta;
    const raw = this.tgtTok.decode(out.map((i) => (i < tn ? i : unk)), { skip_special_tokens: true });
    return postprocess(raw, tgt, pre.map);
  }
}

/** Routes (src, tgt) to a bundle, loading each bundle on first use. */
export class Translator {
  constructor(rt) {
    this.rt = rt;
    this.bundles = new Map();
  }
  async bundle(src, tgt) {
    const name = bundleName(src, tgt);
    if (!this.bundles.has(name)) this.bundles.set(name, IndicTrans.load(this.rt, name));
    return this.bundles.get(name);
  }
  async translate(text, src, tgt) {
    if (src === tgt || !text.trim()) return { text, ms: 0 };
    const t = performance.now();
    const out = await (await this.bundle(src, tgt)).translate(text, src, tgt);
    return { text: out, ms: performance.now() - t };
  }
}
