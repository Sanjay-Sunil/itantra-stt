// Page: setup + model loading, three views (Try it, Language test, How it works), compute panel.
import { LANGUAGES, requiredModels } from "./lib/engine.js";
import { compose, loadClips, refFor } from "./ui/samples.js";
import { renderLidTest, LANG_COLOR } from "./ui/langtest.js";
import { renderHow } from "./ui/how.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
// site.json (optional, written by build.mjs --models-url) points a hosted copy at its model repository
const [manifest, compute, site] = await Promise.all([fetch("models.json").then((r) => r.json()),
  fetch("compute.json").then((r) => r.json()), fetch("site.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}))]);
const MODEL_BASE = params.get("models") || site.modelBase || "./models/";
const ALL = Object.keys(LANGUAGES);
// language test: every language is a candidate, with the speech-model tier chosen in the setup
const lidCfg = () => ({ languages: ALL, target: null, asrMode: "ctc", asrTier: cfgLoaded.asrTier });

const isolated = self.crossOriginIsolated;
const cores = navigator.hardwareConcurrency || 4;
const mb = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const names = (ls) => ls.map((l) => LANGUAGES[l]).join(" · ");

// one quiet status line: how it runs here, and whether the network is still needed
function pills(extra = []) {
  $("#pills").textContent = [isolated ? `WebAssembly, ${$("#threads")?.value || Math.min(cores, 8)} threads` : "WebAssembly, 1 thread",
    ...extra].join(" · ");
}
pills();

// ---------------------------------------------------------------- setup
const defaults = (params.get("langs") || "hi,en").split(",");
$("#langs").innerHTML = Object.entries(LANGUAGES).map(([k, v]) =>
  `<label class="lang"><input type="checkbox" value="${k}" ${defaults.includes(k) ? "checked" : ""}><span>${v}</span></label>`).join("");
$("#target").innerHTML = `<option value="">No translation</option>` + Object.entries(LANGUAGES)
  .map(([k, v]) => `<option value="${k}" ${k === (params.get("target") ?? "en") ? "selected" : ""}>${v}</option>`).join("");
const maxThreads = isolated ? Math.min(cores, 16) : 1;
$("#threads").innerHTML = Array.from({ length: maxThreads }, (_, i) => i + 1)
  .map((n) => `<option ${n === Math.min(maxThreads, 8) ? "selected" : ""}>${n}</option>`).join("");

const config = () => ({ languages: [...document.querySelectorAll("#langs input:checked")].map((i) => i.value),
  target: $("#target").value || null, asrMode: $("#mode").value, asrTier: $("#tier").value });

function neededFiles(cfg) {
  const files = [];
  for (const dir of requiredModels(cfg))
    for (const [f, size] of Object.entries(manifest[dir])) {
      const head = /joint_post_net_(\w+)\.onnx$/.exec(f); // RNN-T heads: only the chosen languages, none for CTC
      if (head && (cfg.asrMode === "ctc" || !cfg.languages.includes(head[1]))) continue;
      files.push({ path: `${dir}/${f}`, dir, size });
    }
  return files;
}
const dirSize = Object.fromEntries(Object.entries(manifest).map(([d, fs]) =>
  [d, Object.entries(fs).filter(([f]) => !/joint_post_net_(?!hi)/.test(f)).reduce((a, [, s]) => a + s, 0)]));

// speech-model tiers (sizes for the chosen languages' heads are close enough: shown per tier)
const tierSize = (t) => Object.values(manifest[`indic-conformer-${t}`] || {}).reduce((a, b) => a + b, 0);
$("#tier").innerHTML = [["int8", "INT8 · fastest"], ["int4mixed", "INT4-mixed · smaller download"]]
  .filter(([t]) => manifest[`indic-conformer-${t}`])
  .map(([t, n]) => `<option value="${t}" ${t === (params.get("tier") || "int8") ? "selected" : ""}>${n} (${mb(tierSize(t))})</option>`).join("");

let cachedUrls = new Set();
try { cachedUrls = new Set((await (await caches.open("itantra-models-v1")).keys()).map((r) => r.url)); } catch { /* no cache */ }

function showDownload() {
  const cfg = config();
  $("#load").disabled = !cfg.languages.length;
  if (!cfg.languages.length) { $("#download").textContent = "Pick at least one language."; return; }
  const files = neededFiles(cfg);
  const total = files.reduce((a, f) => a + f.size, 0);
  const cached = files.filter((f) => cachedUrls.has(new URL(MODEL_BASE + f.path, location.href).href)).reduce((a, f) => a + f.size, 0);
  $("#download").innerHTML = `Download <b>${mb(total)}</b>` + (cached ? ` (${mb(cached)} already cached)` : "");
}
document.querySelectorAll("#langs input, #target, #mode, #tier").forEach((el) => el.addEventListener("change", showDownload));
showDownload();
// phones and low-memory devices: the full model set may not fit in a browser tab
if ((navigator.deviceMemory && navigator.deviceMemory < 8) || /Android|iPhone|iPad/i.test(navigator.userAgent))
  $("#device-note").textContent = "This device may run out of memory with many languages: pick one or two, and use the CTC decoder.";

// ---------------------------------------------------------------- tabs
let tab = "try";
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  tab = b.dataset.tab;
  document.querySelectorAll(".tab").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `panel-${tab}`));
  if (tab === "how") drawHow();
}));

// ---------------------------------------------------------------- worker + jobs
const worker = new Worker("worker.js", { type: "module" });
let cfgLoaded = null, totalBytes = 0, loadedBytes = {}, sessions = [], loadMs = 0, lastStats = null, readyAt = 0, mainRequests = 0;
const segStats = { audioMs: 0, computeMs: 0 };
const allSegs = [];
let job = null; // { name: "try"|"lid", refs, segs, total, started, done }

$("#load").addEventListener("click", () => {
  cfgLoaded = config();
  totalBytes = neededFiles(cfgLoaded).reduce((a, f) => a + f.size, 0);
  document.querySelectorAll("#setup input, #setup select, #load").forEach((el) => (el.disabled = true));
  $("#loading").hidden = false;
  $("#loadtext").textContent = "starting…";
  worker.postMessage({ type: "load", cfg: cfgLoaded, modelBase: MODEL_BASE, manifest, compute, threads: Number($("#threads").value) });
});

function status(text) {
  (job?.name === "lid" ? $("#lid-status") : $("#vadstate")).textContent = text;
}

worker.onmessage = ({ data: m }) => {
  if (m.type === "progress") {
    loadedBytes[m.file] = m.loaded;
    const done = Object.values(loadedBytes).reduce((a, b) => a + b, 0);
    if (!readyAt) {
      $("#loadbar").style.width = `${Math.min(100, (100 * done) / totalBytes)}%`;
      $("#loadtext").textContent = `${mb(done)} / ${mb(totalBytes)} · ${m.fromCache ? "cache" : "downloading"} ${m.file}`;
    } else status(`fetching ${m.file} · ${mb(m.loaded)}`);
  } else if (m.type === "stage") {
    if (!readyAt) $("#loadtext").textContent = `loading ${m.stage} model…`;
  } else if (m.type === "ready") {
    ({ sessions, loadMs } = m);
    readyAt = performance.now();
    lastStats = m.stats;
    document.querySelector("#loading .bar").hidden = true;
    $("#loadtext").textContent = `Ready · ${names(cfgLoaded.languages)}${cfgLoaded.target ? ` → ${LANGUAGES[cfgLoaded.target]}` : ""} · ${cfgLoaded.asrTier === "int4mixed" ? "INT4-mixed" : "INT8"} · ${sessions.length} models in ${(loadMs / 1000).toFixed(1)} s`;
    document.querySelectorAll(".needs-models").forEach((c) => c.setAttribute("aria-disabled", "false"));
    document.querySelectorAll(".needs-models button, .needs-models input").forEach((el) => (el.disabled = false));
    $("#try-sample").textContent = `Play a sample: ${names(cfgLoaded.languages)}`;
    $("#lid-mine").textContent = `Sample: ${names(cfgLoaded.languages)}`;
    $("#load").textContent = "Loaded";
    $("#load").classList.replace("primary", "ghost");
    onlinePill();
    renderStats();
  } else if (m.type === "started") {
    sessions = m.sessions;
    job?.started?.();
  } else if (m.type === "vad") {
    $("#vadbar").style.width = `${Math.round(m.prob * 100)}%`;
    $("#vadbar").classList.toggle("speech", m.triggered);
    if (job?.name === "try") status(m.triggered ? "speech" : "listening");
  } else if (m.type === "queued") {
    status(`${m.queued} part${m.queued > 1 ? "s" : ""} queued`);
  } else if (m.type === "segment") {
    lastStats = m.stats;
    segStats.audioMs += m.seg.durationMs;
    segStats.computeMs += m.seg.timings.total;
    allSegs.push(m.seg);
    if (m.job === "try") addSegment(m.seg);
    else if (job) { job.segs.push(m.seg); status(`${job.segs.length} chunks…`); }
    renderStats();
    if (tab === "how") drawHow();
  } else if (m.type === "flushed") {
    lastStats = m.stats;
    renderStats();
    if (m.job === "lid" && job) {
      renderLidTest($("#lid-results"), { segs: job.segs, refs: job.refs, trace: m.trace, total: job.total || m.trace.length * 0.032 });
      status("done");
    } else status("done");
    job?.done?.();
  } else if (m.type === "error") {
    status(`error: ${m.message}`);
    if (!readyAt) $("#loadtext").textContent = `error: ${m.message}`;
    console.error(m.message);
  }
};

/** Run one audio buffer through a view's configuration. */
async function run(name, cfg, audio, { refs = null, url = null } = {}) {
  job = { name, refs, segs: [], total: audio.length / 16000 };
  const player = name === "lid" ? $("#lid-player") : $("#player");
  if (url) { player.src = url; player.hidden = false; }
  if (name === "try") { $("#segments").innerHTML = ""; tryRefs = refs; }
  else $("#lid-results").innerHTML = `<p class="empty">Analysing…</p>`;
  await new Promise((r) => { job.started = r; worker.postMessage({ type: "start", job: name, cfg }); });
  sendAudio(audio);
  await new Promise((r) => { job.done = r; worker.postMessage({ type: "flush" }); });
}

// ---------------------------------------------------------------- Try it
let tryRefs = null;
const STAGE_COLORS = [["vad", "--s-vad", "VAD"], ["lid", "--s-lid", "LID"], ["lid_ctc", "--s-lid", "check"],
  ["encoder", "--s-enc", "encoder"], ["decode", "--s-dec", "decode"], ["mt", "--s-mt", "translate"]];
const fmtT = (ms) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;

function addSegment(s) {
  const t = s.timings, parts = STAGE_COLORS.filter(([k]) => t[k] > 0), sum = parts.reduce((a, [k]) => a + t[k], 0) || 1;
  const ref = refFor(s, tryRefs);
  const el = document.createElement("div");
  el.className = "seg";
  el.innerHTML = `
    <div class="meta"><span class="chip" style="--c:${LANG_COLOR[s.language]}">${LANGUAGES[s.language] || s.language}</span>
      <span>${fmtT(s.startMs)}–${fmtT(s.endMs)}</span>
      <span>${(t.total / 1000).toFixed(2)} s · ${(s.macs / 1e9).toFixed(1)} GMAC</span>
      ${ref && ref.language !== s.language ? `<span class="flag">actual: ${LANGUAGES[ref.language]}</span>` : ""}
      ${s.flags.map((f) => `<span class="flag">${f.replace("_", " ")}</span>`).join("")}</div>
    <div class="text" lang="${s.language}">${esc(s.text || "—")}</div>
    ${s.translation ? `<div class="tr" lang="${s.targetLanguage}">${esc(s.translation)}</div>` : ""}
    ${ref ? `<div class="ref" lang="${ref.language}"><span>said</span> ${esc(ref.text)}</div>` : ""}
    <div class="tbar" title="${parts.map(([k, , n]) => `${n} ${t[k].toFixed(0)} ms`).join(" · ")}">${parts.map(([k, c]) =>
      `<i style="width:${(100 * t[k]) / sum}%;background:var(${c})"></i>`).join("")}</div>`;
  $("#segments").append(el);
  el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

$("#try-sample").addEventListener("click", async () => {
  const s = await compose(cfgLoaded.languages);
  if (!s) return status("no sample clips: run npm run build");
  run("try", cfgLoaded, s.audio, s);
});
loadClips();

// ---------------------------------------------------------------- Language test
$("#lid-all").addEventListener("click", async () => { const s = await compose(ALL, { maxSeconds: 0 }); run("lid", lidCfg(), s.audio, s); });
$("#lid-mine").addEventListener("click", async () => { const s = await compose(cfgLoaded.languages); run("lid", lidCfg(), s.audio, s); });

// ---------------------------------------------------------------- How it works
function drawHow() {
  renderHow($("#how-body"), { segs: allSegs, rs: lastStats?.runStats || {}, size: dirSize, threads: $("#threads").value,
    tier: cfgLoaded?.asrTier || $("#tier").value,
    target: cfgLoaded?.target ? LANGUAGES[cfgLoaded.target] : null });
}
drawHow();

// ---------------------------------------------------------------- compute panel
const LABELS = [["vad", "Voice activity"], ["lid", "Language ID"], ["encoder", "Speech encoder"], ["ctc", "CTC head"],
  ["rnnt_pred", "RNN-T prediction"], ["rnnt_joint", "RNN-T joint"], ["whisper_enc", "Whisper encoder"],
  ["whisper_dec", "Whisper decoder"], ["mt", "Translation"]];

function renderStats() {
  if (!lastStats) return;
  const rs = lastStats.runStats;
  let gmac = 0;
  $("#models tbody").innerHTML = LABELS.filter(([k]) => rs[k]?.runs).map(([k, n]) => {
    gmac += rs[k].macs / 1e9;
    return `<tr><td>${n}</td><td>${(rs[k].ms / 1000).toFixed(2)} s</td><td>${(rs[k].macs / 1e9).toFixed(1)}</td></tr>`;
  }).join("");
  $("#k-audio").textContent = `${(segStats.audioMs / 1000).toFixed(1)} s`;
  $("#k-rtf").textContent = segStats.audioMs ? `${(segStats.computeMs / segStats.audioMs).toFixed(2)}×` : "–";
  $("#k-gmac").textContent = gmac.toFixed(gmac < 100 ? 1 : 0);
  const net = lastStats.net, reqs = net.requestsAfterReady + mainRequests;
  $("#k-net").textContent = String(reqs);
  $("#k-net").classList.toggle("zero", reqs === 0);
  $("#facts").innerHTML = [["Models", `${sessions.length} · ${mb(sessions.reduce((a, s) => a + s.bytes, 0))}`],
    ["Downloaded / cached", `${mb(net.downloaded)} / ${mb(net.cached)}`], ["Threads", $("#threads").value]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// requests the page makes after the models are ready (sample clips excluded: they are the input)
new PerformanceObserver((list) => {
  for (const e of list.getEntries()) if (readyAt && e.startTime > readyAt && !e.name.includes("/samples/")) mainRequests++;
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

async function runFile(name, arrayBuffer, url) {
  run(name, name === "lid" ? lidCfg() : cfgLoaded, await decodeTo16k(arrayBuffer), { url });
}
$("#file").addEventListener("change", async (e) => e.target.files[0] && runFile("try", await e.target.files[0].arrayBuffer(), URL.createObjectURL(e.target.files[0])));
$("#lid-file").addEventListener("change", async (e) => e.target.files[0] && runFile("lid", await e.target.files[0].arrayBuffer(), URL.createObjectURL(e.target.files[0])));

// microphone → AudioWorklet → 16 kHz chunks → worker
const WORKLET = `class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) { const ch = inputs[0][0]; if (ch) { this.buf.push(ch.slice()); this.n += ch.length;
    if (this.n >= 2048) { const out = new Float32Array(this.n); let o = 0; for (const b of this.buf) { out.set(b, o); o += b.length; }
      this.port.postMessage(out, [out.buffer]); this.buf = []; this.n = 0; } } return true; } }
registerProcessor("tap", Tap);`;
let mic = null;

function resampler(from) {
  // box-filtered decimation for browsers that cannot open a 16 kHz AudioContext
  if (from === 16000) return (x) => x;
  const ratio = from / 16000, w = Math.max(1, Math.round(ratio));
  let pos = 0, prev = new Float32Array(0);
  return (x) => {
    const buf = new Float32Array(prev.length + x.length);
    buf.set(prev);
    buf.set(x, prev.length);
    const out = [];
    for (; pos + w < buf.length; pos += ratio) {
      let s = 0;
      for (let k = 0; k < w; k++) s += buf[Math.floor(pos) + k];
      out.push(s / w);
    }
    const keep = Math.floor(pos);
    prev = buf.slice(keep);
    pos -= keep;
    return Float32Array.from(out);
  };
}

async function toggleMic(name, button) {
  if (mic) {
    mic.stream.getTracks().forEach((t) => t.stop());
    mic.ctx.close();
    const which = mic.button;
    mic = null;
    which.textContent = which.dataset.label;
    which.classList.remove("recording");
    job.total = null;
    await new Promise((r) => { job.done = r; worker.postMessage({ type: "flush" }); });
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  job = { name, refs: null, segs: [], total: null };
  if (name === "try") { $("#segments").innerHTML = ""; tryRefs = null; }
  await new Promise((r) => { job.started = r; worker.postMessage({ type: "start", job: name, cfg: name === "lid" ? lidCfg() : cfgLoaded }); });
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
  button.dataset.label ||= button.textContent;
  mic = { stream, ctx, button };
  button.textContent = "Stop";
  button.classList.add("recording");
  status("listening");
}
$("#mic").addEventListener("click", () => toggleMic("try", $("#mic")));
$("#lid-mic").addEventListener("click", () => toggleMic("lid", $("#lid-mic")));

// offline indicator: once the models are loaded, the page keeps working with the network off
function onlinePill() {
  if (readyAt) pills([navigator.onLine ? "models loaded, works offline now" : "offline, still working"]);
}
addEventListener("online", onlinePill);
addEventListener("offline", onlinePill);
window.itantra = { run, compose, cfg: () => cfgLoaded }; // test hook (test/browser_e2e.mjs)
