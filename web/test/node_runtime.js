// Runtime for Node tests: models read from the repository's models/ directory
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";
import { Tokenizer } from "@huggingface/tokenizers";
import { Runtime } from "../lib/runtime.js";

export const MODELS = process.env.INDICSTT_MODELS_DIR || path.resolve(import.meta.dirname, "../../models");

export function nodeRuntime(threads = 4) {
  ort.env.wasm.numThreads = threads;
  return new Runtime({ ort, Tokenizer, fetchBytes: async (p) => new Uint8Array(fs.readFileSync(path.join(MODELS, p))) });
}

export function readWav16k(file) {
  const b = fs.readFileSync(file);
  // minimal PCM16 mono WAV reader (the smoke/demo sets are 16 kHz mono)
  let off = 12, fmt = null;
  while (off < b.length) {
    const id = b.toString("ascii", off, off + 4), size = b.readUInt32LE(off + 4);
    if (id === "fmt ") fmt = { ch: b.readUInt16LE(off + 10), sr: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === "data") {
      if (!fmt || fmt.sr !== 16000 || fmt.ch !== 1 || fmt.bits !== 16) throw new Error(`${file}: need 16 kHz mono PCM16`);
      const n = size / 2, out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = b.readInt16LE(off + 8 + 2 * i) / 32768;
      return out;
    }
    off += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}
