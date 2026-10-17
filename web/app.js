// Page: setup, model download progress, audio input (microphone / file), transcript and compute panel.
import { LANGUAGES, requiredModels } from "./lib/engine.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("models") || "./models/";
const [manifest, compute] = await Promise.all([fetch("models.json").then((r) => r.json()),
  fetch("compute.json").then((r) => r.json())]);

const isolated = self.crossOriginIsolated;
const cores = navigator.hardwareConcurrency || 4;
const simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10,
  1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));

// ---------------------------------------------------------------- environment pills
function pills(extra = []) {
  const p = [
    [isolated ? "ok" : "warn", isolated ? "Multi-threaded WebAssembly" : "Single-threaded (no cross-origin isolation)"],
    [simd ? "ok" : "warn", simd ? "WASM SIMD" : "No WASM SIMD"],
    ["", `${cores} logical cores`],
    ...extra,
  ];
  $("#pills").innerHTML = p.map(([c, t]) => `<span class="pill ${c}">${t}</span>`).join("");
}
pills();

// ---------------------------------------------------------------- setup
const defaults = (params.get("langs") || "hi,en").split(",");
$("#langs").innerHTML = Object.entries(LANGUAGES).map(([k, v]) =>
  `<label class="lang"><input type="checkbox" value="${k}" ${defaults.includes(k) ? "checked" : ""}><span>${v}</span></label>`).join("");
$("#target").innerHTML = `<option value="">No translation</option>` +
  Object.entries(LANGUAGES).map(([k, v]) => `<option value="${k}" ${k === (params.get("target") ?? "en") ? "selected" : ""}>${v}</option>`).join("");
const maxThreads = isolated ? Math.min(cores, 16) : 1;
$("#threads").innerHTML = Array.from({ length: maxThreads }, (_, i) => i + 1)
  .map((n) => `<option ${n === Math.min(maxThreads, 8) ? "selected" : ""}>${n}</option>`).join("");

function config() {
  const languages = [...document.querySelectorAll("#langs input:checked")].map((i) => i.value);
  return { languages, target: $("#target").value || null, asrMode: $("#mode").value };
}

// files the configuration will fetch: per-language RNN-T heads only for the chosen languages
function neededFiles(cfg) {
  const files = [];
  for (const dir of requiredModels(cfg))
    for (const [f, size] of Object.entries(manifest[dir])) {
      const head = /joint_post_net_(\w+)\.onnx$/.exec(f);
      if (head && !cfg.languages.includes(head[1])) continue;
      files.push({ path: `${dir}/${f}`, dir, size });
    }
  return files;
}

const MODEL_NAMES = { "silero-vad": "Voice activity", "lid-ambernet": "Language ID", "indic-conformer-int8": "Indic speech",
  "whisper-tiny-en": "English speech", "mt-indic-en": "Translation Indic→English", "mt-en-indic": "Translation English→Indic",
  "mt-indic-indic": "Translation Indic↔Indic" };
const mb = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;

let cachedUrls = new Set();
async function refreshCache() {
  try {
    const c = await caches.open("itantra-models-v1");
    cachedUrls = new Set((await c.keys()).map((r) => r.url));
  } catch { /* Cache Storage unavailable: everything downloads */ }
}

function showDownload() {
  const cfg = config();
  if (!cfg.languages.length) { $("#download").innerHTML = "Pick at least one language."; $("#load").disabled = true; return; }
  $("#load").disabled = false;
  const files = neededFiles(cfg);
  const per = {};
  let total = 0, cached = 0;
  for (const f of files) {
    per[f.dir] = (per[f.dir] || 0) + f.size;
    total += f.size;
    if (cachedUrls.has(new URL(MODEL_BASE + f.path, location.href).href)) cached += f.size;
  }
  $("#download").innerHTML = `Needs <b>${mb(total)}</b>` + (cached ? ` (<b>${mb(cached)}</b> already in this browser's cache)` : "") +
    ": " + Object.entries(per).map(([d, s]) => `${MODEL_NAMES[d]} ${mb(s)}`).join(" · ");
}
document.querySelectorAll("#langs input, #target").forEach((el) => el.addEventListener("change", showDownload));
await refreshCache();
showDownload();

// ---------------------------------------------------------------- worker
const worker = new Worker("worker.js", { type: "module" });
let cfgLoaded = null, files = [], loadedBytes = {}, totalBytes = 0, sessions = [], loadMs = 0, lastStats = null;
const segStats = { audioMs: 0, computeMs: 0 };
let readyAt = 0, mainRequests = 0;

$("#load").addEventListener("click", () => {
  cfgLoaded = config();
  files = neededFiles(cfgLoaded);
  totalBytes = files.reduce((a, f) => a + f.size, 0);
  document.querySelectorAll("#setup input, #setup select, #load").forEach((el) => (el.disabled = true));
  $("#loading").hidden = false;
  $("#loadtext").textContent = "starting…";
  worker.postMessage({ type: "load", cfg: cfgLoaded, modelBase: MODEL_BASE, manifest, compute,
    threads: Number($("#threads").value) });
});

const STAGES = { vad: "voice activity model", lid: "language ID model", asr: "Indic speech model (largest)",
  english: "English speech model", mt: "translation model" };

worker.onmessage = ({ data: m }) => {
  if (m.type === "progress") {
    loadedBytes[m.file] = m.loaded;
    const done = Object.values(loadedBytes).reduce((a, b) => a + b, 0);
    $("#loadbar").style.width = `${Math.min(100, (100 * done) / totalBytes)}%`;
    $("#loadtext").textContent = `${mb(done)} / ${mb(totalBytes)} · ${m.fromCache ? "from cache" : "downloading"} ${m.file}`;
  } else if (m.type === "stage") {
    $("#loadtext").textContent = `loading ${STAGES[m.stage]}…`;
  } else if (m.type === "ready") {
    sessions = m.sessions;
    loadMs = m.loadMs;
    readyAt = performance.now();
    $("#loadbar").style.width = "100%";
    $("#loadtext").textContent = `ready: ${sessions.length} ONNX models in ${(loadMs / 1000).toFixed(1)} s`;
    $("#input").setAttribute("aria-disabled", "false");
    document.querySelectorAll("#input button, #file").forEach((el) => (el.disabled = false));
    onlinePill();
    lastStats = m.stats;
    renderStats();
  } else if (m.type === "vad") {
    $("#vadbar").style.width = `${Math.round(m.prob * 100)}%`;
    $("#vadbar").classList.toggle("speech", m.triggered);
    $("#vadstate").textContent = m.triggered ? "speech" : "listening";
  } else if (m.type === "queued") {
    $("#vadstate").textContent = `${m.queued} queued`;
  } else if (m.type === "segment") {
    lastStats = m.stats;
    addSegment(m.seg);
    renderStats();
  } else if (m.type === "flushed") {
    lastStats = m.stats;
    renderStats();
    $("#vadstate").textContent = "done";
  } else if (m.type === "log") {
    console.warn(m.message);
  } else if (m.type === "error") {
    $("#loadtext").textContent = `error: ${m.message}`;
    console.error(m.message);
  }
};

// ---------------------------------------------------------------- transcript
const STAGE_COLORS = [["vad", "--s-vad", "VAD"], ["lid", "--s-lid", "LID"], ["lid_ctc", "--s-lid", "LID check"],
  ["encoder", "--s-enc", "encoder"], ["decode", "--s-dec", "decode"], ["mt", "--s-mt", "translate"]];
const fmtT = (ms) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function addSegment(s) {
  if ($("#segments .empty")) $("#segments").innerHTML = "";
  segStats.audioMs += s.durationMs;
  segStats.computeMs += s.timings.total;
  const t = s.timings;
  const parts = STAGE_COLORS.filter(([k]) => t[k] > 0);
  const sum = parts.reduce((a, [k]) => a + t[k], 0) || 1;
  const el = document.createElement("div");
  el.className = "seg";
  el.innerHTML = `
    <div class="meta"><span class="chip">${LANGUAGES[s.language] || s.language}</span>
      <span>${fmtT(s.startMs)}–${fmtT(s.endMs)}</span>
      <span>${s.decidedBy}${s.confidence < 1 ? ` · ${(100 * s.confidence).toFixed(0)}%` : ""}</span>
      <span>${(s.durationMs / 1000).toFixed(1)} s audio → ${(t.total / 1000).toFixed(2)} s compute · ${(s.macs / 1e9).toFixed(1)} GMAC</span>
      ${s.flags.map((f) => `<span class="flag">${f.replace("_", " ")}</span>`).join("")}</div>
    <div class="text" lang="${s.language}">${esc(s.text || "—")}</div>
    ${s.translation ? `<div class="tr" lang="${s.targetLanguage}">${esc(s.translation)}</div>` : ""}
    <div class="tbar">${parts.map(([k, c]) => `<i style="width:${(100 * t[k]) / sum}%;background:var(${c})" title="${k} ${t[k].toFixed(0)} ms"></i>`).join("")}</div>
    <div class="legend">${parts.map(([k, c, n]) => `<span><i style="background:var(${c})"></i>${n} ${t[k].toFixed(0)} ms</span>`).join("")}</div>`;
  $("#segments").append(el);
  el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------- compute panel
const LABELS = [["vad", "Voice activity · Silero"], ["lid", "Language ID · AmberNet"], ["encoder", "Speech encoder · Conformer 600M"],
  ["ctc", "CTC head"], ["rnnt_pred", "RNN-T prediction net"], ["rnnt_joint", "RNN-T joint net"],
  ["whisper_enc", "Whisper encoder"], ["whisper_dec", "Whisper decoder"], ["mt", "Translation · IndicTrans2"]];

async function memoryMB() {
  try {
    if (performance.measureUserAgentSpecificMemory) return (await performance.measureUserAgentSpecificMemory()).bytes / 1e6;
  } catch { /* not available in this browser */ }
  return null;
}

let mem = null;
async function renderStats() {
  if (!lastStats) return;
  const rs = lastStats.runStats;
  let gmac = 0;
  $("#models tbody").innerHTML = LABELS.filter(([k]) => rs[k]).map(([k, name]) => {
    gmac += rs[k].macs / 1e9;
    return `<tr><td>${name}</td><td>${rs[k].runs.toLocaleString()}</td><td>${(rs[k].ms / 1000).toFixed(2)} s</td><td>${(rs[k].macs / 1e9).toFixed(1)}</td></tr>`;
  }).join("");
  const modelMs = Object.values(rs).reduce((a, s) => a + s.ms, 0);
  $("#k-audio").textContent = `${(segStats.audioMs / 1000).toFixed(1)} s`;
  $("#k-time").textContent = `${(segStats.computeMs / 1000).toFixed(1)} s`;
  $("#k-rtf").textContent = segStats.audioMs ? (segStats.computeMs / segStats.audioMs).toFixed(2) : "–";
  $("#k-gmac").textContent = gmac.toFixed(gmac < 100 ? 1 : 0);
  const net = lastStats.net;
  const reqs = net.requestsAfterReady + mainRequests;
  const modelBytes = sessions.reduce((a, s) => a + s.bytes, 0);
  const rows = [
    ["Throughput", modelMs ? `${(gmac / (modelMs / 1000)).toFixed(1)} GMAC/s` : "–"],
    ["Models in memory", `${sessions.length} ONNX graphs · ${mb(modelBytes)}`],
    ["Model load", `${(loadMs / 1000).toFixed(1)} s`],
    ["Downloaded this visit", mb(net.downloaded)],
    ["From browser cache", mb(net.cached)],
    ["WASM threads", $("#threads").value],
    ["Tab memory", mem ? mb(mem * 1e6) : "n/a in this browser"],
    ["Requests sent while transcribing", `<span class="${reqs ? "" : "zero"}">${reqs}</span>`],
  ];
  $("#facts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}
setInterval(async () => { if (lastStats) { mem = await memoryMB(); renderStats(); } }, 5000);

// count every network request the page itself makes after the models are ready (sample clips excluded: they are input)
new PerformanceObserver((list) => {
  for (const e of list.getEntries())
    if (readyAt && e.startTime > readyAt && !e.name.includes("/samples/")) mainRequests++;
}).observe({ type: "resource", buffered: false });

// ---------------------------------------------------------------- audio input
const CHUNK = 4096;
function sendAudio(f32) {
  for (let i = 0; i < f32.length; i += CHUNK) {
    const c = f32.slice(i, i + CHUNK);
    worker.postMessage({ type: "audio", chunk: c }, [c.buffer]);
  }
}

async function decodeTo16k(buf) {
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(buf);
  tmp.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

async function runFile(arrayBuffer, url, langs) {
  if (langs && cfgLoaded) {
    const missing = langs.filter((l) => !cfgLoaded.languages.includes(l));
    if (missing.length) $("#vadstate").textContent = `note: sample has ${missing.join(", ")}`;
  }
  $("#player").hidden = false;
  $("#player").src = url;
  const audio = await decodeTo16k(arrayBuffer);
  sendAudio(audio);
  worker.postMessage({ type: "flush" });
}

$("#file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) runFile(await f.arrayBuffer(), URL.createObjectURL(f));
});
document.querySelectorAll(".sample").forEach((b) => b.addEventListener("click", async () => {
  const r = await fetch(b.dataset.src);
  if (!r.ok) { $("#vadstate").textContent = "sample missing: run node build.mjs"; return; }
  runFile(await r.arrayBuffer(), b.dataset.src, b.dataset.langs.split(","));
}));

// microphone → AudioWorklet → 16 kHz chunks → worker
const WORKLET = `class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) { const ch = inputs[0][0]; if (ch) { this.buf.push(ch.slice()); this.n += ch.length;
    if (this.n >= 2048) { const out = new Float32Array(this.n); let o = 0; for (const b of this.buf) { out.set(b, o); o += b.length; }
      this.port.postMessage(out, [out.buffer]); this.buf = []; this.n = 0; } } return true; } }
registerProcessor("tap", Tap);`;
let mic = null;

function resampler(from) {
  // linear-interpolation resampler with a box pre-filter, for browsers that cannot open a 16 kHz AudioContext
  if (from === 16000) return (x) => x;
  const ratio = from / 16000;
  let pos = 0, prev = new Float32Array(0);
  return (x) => {
    const buf = new Float32Array(prev.length + x.length);
    buf.set(prev);
    buf.set(x, prev.length);
    const out = [];
    const w = Math.max(1, Math.round(ratio));
    for (; pos + w < buf.length; pos += ratio) {
      let s = 0;
      const i0 = Math.floor(pos);
      for (let k = 0; k < w; k++) s += buf[i0 + k];
      out.push(s / w);
    }
    const keep = Math.floor(pos);
    prev = buf.slice(keep);
    pos -= keep;
    return Float32Array.from(out);
  };
}

$("#mic").addEventListener("click", async () => {
  if (mic) {
    mic.stream.getTracks().forEach((t) => t.stop());
    mic.ctx.close();
    mic = null;
    $("#mic").textContent = "🎙 Start microphone";
    $("#mic").classList.remove("recording");
    worker.postMessage({ type: "flush" });
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  let ctx;
  try { ctx = new AudioContext({ sampleRate: 16000 }); } catch { ctx = new AudioContext(); }
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
  await ctx.audioWorklet.addModule(url);
  let src;
  try { src = ctx.createMediaStreamSource(stream); } catch {
    ctx.close();
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule(url);
    src = ctx.createMediaStreamSource(stream);
  }
  const node = new AudioWorkletNode(ctx, "tap");
  const rs = resampler(ctx.sampleRate);
  node.port.onmessage = ({ data }) => sendAudio(rs(data));
  src.connect(node);
  mic = { stream, ctx };
  $("#mic").textContent = "■ Stop microphone";
  $("#mic").classList.add("recording");
  $("#vadstate").textContent = "listening";
});

// offline indicator: once the models are loaded, the page keeps working with the network off
function onlinePill() {
  if (!readyAt) return;
  pills([["ok", `Models loaded · ${cfgLoaded.languages.map((l) => LANGUAGES[l]).join(", ")}`],
    navigator.onLine ? ["", "Online: you can disconnect now"] : ["ok", "Offline: still transcribing"]]);
}
addEventListener("online", onlinePill);
addEventListener("offline", onlinePill);
window.itantra = { runFile }; // test hook (test/browser_e2e.mjs OFFLINE=1)
