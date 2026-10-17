// English ASR: whisper-tiny.en INT8, ported from src/indicstt/english.py

import { whisperLogMel } from "./features.js";

const SOT = 50257, NO_TIMESTAMPS = 50362, EOT = 50256;

export class WhisperEnglish {
  static async load(rt) {
    const m = new WhisperEnglish();
    m.rt = rt;
    m.enc = await rt.session("whisper-tiny-en/onnx/encoder_model_int8.onnx", { arena: false });
    m.dec = await rt.session("whisper-tiny-en/onnx/decoder_model_merged_int8.onnx");
    m.tok = new rt.Tokenizer(await rt.json("whisper-tiny-en/tokenizer.json"),
      await rt.json("whisper-tiny-en/tokenizer_config.json"));
    m.pastNames = m.dec.inputNames.filter((n) => n.startsWith("past_key_values"));
    return m;
  }

  async transcribe(audio, maxTokens = 200) {
    const rt = this.rt;
    const t0 = performance.now();
    const hidden = (await rt.run("whisper_enc", this.enc,
      { input_features: rt.f32(whisperLogMel(audio), [1, 80, 3000]) })).last_hidden_state;
    rt.count("whisper_enc", "whisper_enc");
    const t1 = performance.now();
    const empty = rt.f32(new Float32Array(0), [1, 6, 0, 64]);
    const past = Object.fromEntries(this.pastNames.map((n) => [n, empty]));
    let cur = [SOT, NO_TIMESTAMPS];
    const out = [];
    for (let step = 0; step < maxTokens; step++) {
      const res = await rt.run("whisper_dec", this.dec, {
        input_ids: rt.i64(cur, [1, cur.length]), encoder_hidden_states: hidden,
        use_cache_branch: new rt.ort.Tensor("bool", [step > 0], [1]), ...past });
      rt.count("whisper_dec", "whisper_dec", step + 2);
      const lg = res.logits, V = lg.dims[2], off = (lg.dims[1] - 1) * V;
      let nxt = 0;
      for (let k = 1; k < V; k++) if (lg.data[off + k] > lg.data[off + nxt]) nxt = k;
      if (nxt === EOT) break;
      out.push(nxt);
      for (const name of this.pastNames) {
        // encoder KV come back empty on cached steps; keep the first ones
        if (step === 0 || name.includes(".decoder.")) past[name] = res[name.replace("past_key_values", "present")];
      }
      cur = [nxt];
    }
    const text = this.tok.decode(out, { skip_special_tokens: true }).trim();
    return { text, encoderMs: t1 - t0, decodeMs: performance.now() - t1 };
  }
}
