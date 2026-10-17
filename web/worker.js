// Engine worker: downloads (or reads from Cache Storage) the models, runs the pipeline, reports compute.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Runtime } from "./lib/runtime.js";
import { Engine } from "./lib/engine.js";

const CACHE = "itantra-models-v1";
let engine = null, rt = null, job = null;
let audioChain = Promise.resolve();
// audio, start and flush run strictly in order; a failure is reported and the chain keeps going
const chain = (fn) => (audioChain = audioChain.then(fn).catch((e) => post({ type: "error", message: String(e?.message || e) })));
const pool = {}; // every model loaded so far, shared by the configurations of the different views
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
      cache.put(url, new Response(b, { headers: resp.headers })).catch((e) => post({ type: "log", message: `cache ${p}: ${e}` }));
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
      // time-to-text: part closed → text ready, including any wait behind earlier parts
      post({ type: "segment", job, seg: { ...rest, durationMs: (audio.length / 16000) * 1000, macs: after - before,
        timeToText: performance.now() - seg.finalizedAt },
        queued: queue.length, stats: stats() });
    }
  } catch (e) {
    post({ type: "error", message: String(e?.message || e) });
  }
}

async function onAudio(chunk) {
  const segs = await engine.push(chunk);
  for (const s of segs) s.finalizedAt = performance.now(); // voice detection closed the part (0.4 s after speech ended)
  post({ type: "vad", prob: engine.segmenter.lastProb, triggered: engine.segmenter.triggered });
  if (segs.length) { queue.push(...segs); post({ type: "queued", queued: queue.length }); drain(); }
}

async function useConfig(cfg) {
  const t = performance.now();
  engine = await Engine.load(rt, cfg, (stage) => post({ type: "stage", stage }), pool);
  for (const k of ["vad", "lid", "indic", "english", "mt"]) if (engine[k]) pool[k] = engine[k];
  return performance.now() - t;
}

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "load") {
      ort.env.wasm.numThreads = m.threads;
      ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
      rt = new Runtime({ ort, Tokenizer, fetchBytes: makeFetch(m.modelBase, m.manifest) });
      rt.compute = m.compute;
      const loadMs = await useConfig(m.cfg);
      ready = true;
      post({ type: "ready", loadMs, sessions: rt.sessions, stats: stats() });
    } else if (m.type === "start") {
      // a new stream for one view ("try" or "lid"), possibly with other languages / options
      chain(async () => {
        await drain();
        job = m.job;
        ready = false; // models fetched for a new configuration are downloads, not transcription traffic
        const extraMs = await useConfig(m.cfg);
        ready = true;
        post({ type: "started", job, extraMs, sessions: rt.sessions, stats: stats() });
      });
    } else if (m.type === "audio") {
      chain(() => onAudio(m.chunk));
    } else if (m.type === "flush") {
      chain(async () => {
        const rest = engine.flush();
        for (const s of rest) s.finalizedAt = performance.now();
        queue.push(...rest);
        await drain();
        const trace = Float32Array.from(engine.vadTrace);
        engine.resetStream();
        post({ type: "flushed", job, trace, stats: stats() }, [trace.buffer]);
      });
    }
  } catch (e) {
    post({ type: "error", message: String(e?.message || e) });
  }
};
