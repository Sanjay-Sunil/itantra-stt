// Engine worker: downloads (or reads from Cache Storage) the models, runs the pipeline, reports compute.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Runtime } from "./lib/runtime.js";
import { Engine } from "./lib/engine.js";

const CACHE = "itantra-models-v1";
let engine = null, rt = null;
const queue = [];
let draining = null;
const net = { downloaded: 0, cached: 0, requestsAfterReady: 0 };
let ready = false;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function openCache() {
  try { return await caches.open(CACHE); } catch { return null; }
}

function makeFetch(base, manifest) {
  return async (p) => {
    const url = new URL(base + p, self.location.href).href;
    if (ready) net.requestsAfterReady++;
    const cache = await openCache();
    let resp = cache ? await cache.match(url) : null;
    const fromCache = !!resp;
    if (!resp) {
      resp = await fetch(url);
      if (!resp.ok) throw new Error(`${p}: HTTP ${resp.status}`);
    }
    const [dir, ...rest] = p.split("/");
    const total = Number(resp.headers.get("Content-Length")) || manifest[dir]?.[rest.join("/")] || 0;
    let body = resp.body;
    if (!fromCache && cache) {
      const [a, b] = body.tee();
      body = a;
      cache.put(url, new Response(b, { headers: resp.headers })).catch(() => {});
    }
    const out = total ? new Uint8Array(total) : null;
    const chunks = [];
    let loaded = 0, last = 0;
    const reader = body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (out && loaded + value.length <= total) out.set(value, loaded);
      else chunks.push(value);
      loaded += value.length;
      if (loaded - last > 4e6 || loaded === total) {
        last = loaded;
        post({ type: "progress", file: p, loaded, total, fromCache });
      }
    }
    post({ type: "progress", file: p, loaded, total: loaded, fromCache, done: true });
    if (fromCache) net.cached += loaded; else net.downloaded += loaded;
    if (out && loaded === total && !chunks.length) return out;
    const all = new Uint8Array(loaded);
    let o = 0;
    if (out) { all.set(out.subarray(0, Math.min(loaded, total))); o = Math.min(loaded, total); }
    for (const c of chunks) { all.set(c, o); o += c.length; }
    return all;
  };
}

function stats() {
  return { runStats: Object.fromEntries(rt.runStats), net, sessions: rt.sessions.length };
}

// One processing loop at a time; segments queued while it runs are picked up by the same loop.
function drain() {
  if (!draining) draining = processQueue().finally(() => (draining = null));
  return draining;
}

async function processQueue() {
  try {
    while (queue.length) {
      const seg = queue.shift();
      const before = [...rt.runStats.values()].reduce((a, s) => a + s.macs, 0);
      await engine.process(seg);
      const after = [...rt.runStats.values()].reduce((a, s) => a + s.macs, 0);
      const { audio, ...rest } = seg;
      post({ type: "segment", seg: { ...rest, durationMs: (audio.length / 16000) * 1000, macs: after - before },
        queued: queue.length, stats: stats() });
    }
  } catch (e) {
    post({ type: "error", message: String(e?.message || e) });
  }
}

async function onAudio(chunk) {
  const segs = await engine.push(chunk);
  post({ type: "vad", prob: engine.segmenter.lastProb, triggered: engine.segmenter.triggered });
  if (segs.length) { queue.push(...segs); post({ type: "queued", queued: queue.length }); drain(); }
}

let audioChain = Promise.resolve();
self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "load") {
      ort.env.wasm.numThreads = m.threads;
      ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
      rt = new Runtime({ ort, Tokenizer, fetchBytes: makeFetch(m.modelBase, m.manifest) });
      rt.compute = m.compute;
      const t = performance.now();
      engine = await Engine.load(rt, m.cfg, (stage) => post({ type: "stage", stage }));
      ready = true;
      post({ type: "ready", loadMs: performance.now() - t, sessions: rt.sessions, stats: stats() });
    } else if (m.type === "audio") {
      audioChain = audioChain.then(() => onAudio(m.chunk));
    } else if (m.type === "flush") {
      audioChain = audioChain.then(() => {
        const segs = engine.flush();
        queue.push(...segs);
        engine.resetStream();
        return drain().then(() => post({ type: "flushed", stats: stats() }));
      });
    }
  } catch (e) {
    post({ type: "error", message: String(e?.message || e) });
  }
};
