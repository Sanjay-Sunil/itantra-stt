// End-to-end test in a real (headless) Chromium: load the page, load models, run a sample clip, read results.
//   node serve.mjs &   node test/browser_e2e.mjs [url]     (PROFILE=dir reuses a browser profile; OFFLINE=1; SHOT_DIR=dir)
// Uses the DevTools protocol directly (Node's built-in WebSocket), so no extra dependencies.
import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

const URL_ = process.argv[2] || "http://127.0.0.1:8080/?langs=en,hi,ta";
const CHROME = process.env.CHROME || "/usr/bin/chromium";
const profile = process.env.PROFILE || fs.mkdtempSync(path.join(os.tmpdir(), "itantra-e2e-")); // reuse to test the model cache
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${profile}`,
  "--no-first-run", "--autoplay-policy=no-user-gesture-required",
  // MIC=<wav>: the microphone path, fed from a file through Chromium's fake capture device
  ...(process.env.MIC ? ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${process.env.MIC}%noloop`] : []), "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let targets;
for (let i = 0; i < 50 && !targets; i++) {
  await sleep(200);
  targets = await fetch("http://127.0.0.1:9333/json").then((r) => r.json()).catch(() => null);
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled") console.log("[console]", m.params.args.map((a) => a.value ?? a.description).join(" "));
  if (m.method === "Runtime.exceptionThrown") console.log("[exception]", m.params.exceptionDetails.exception?.description);
};
const cmd = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => (await cmd("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const waitFor = async (expr, timeoutS, label) => {
  const t = Date.now();
  while (!(await evaluate(expr))) {
    if (Date.now() - t > timeoutS * 1000) throw new Error(`timeout waiting for ${label}`);
    await sleep(500);
  }
  return (Date.now() - t) / 1000;
};

try {
  await cmd("Runtime.enable");
  await cmd("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cmd("Page.navigate", { url: URL_ });
  await waitFor("!!document.querySelector('#load') && document.querySelector('#download').textContent.length > 0", 30, "page");
  console.log("isolated:", await evaluate("self.crossOriginIsolated"), "|", await evaluate("document.querySelector('#download').textContent"));
  await evaluate("document.querySelector('#load').click()");
  const loadS = await waitFor("/^(Ready|error)/.test(document.querySelector('#loadtext').textContent)", 900, "models");
  console.log(`models: ${await evaluate("document.querySelector('#loadtext').textContent")} (wall ${loadS.toFixed(1)} s)`);
  const shot = async (name) => {
    const r = await cmd("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
    const out = path.join(process.env.SHOT_DIR || os.tmpdir(), `itantra-${name}.png`);
    fs.writeFileSync(out, Buffer.from(r.result.data, "base64"));
    console.log("screenshot:", out);
  };

  // Try it: sample assembled for the selected languages (OFFLINE=1: network cut after loading)
  if (process.env.REALTIME) {
    // latency + memory: stream a clip at speaking pace (32 ms chunks) and time each part from pause to text
    const clip = process.env.REALTIME === "1" ? "samples/mixed_en_hi_ta.wav" : process.env.REALTIME;
    await evaluate(`fetch("${clip}").then((r) => r.arrayBuffer()).then((b) => { window.__a = window.itantra.parseWav(b); return 1; })`);
    const n0 = await evaluate("window.itantra.segs.length");
    await evaluate("window.itantra.run('try', window.itantra.cfg(), window.__a, { realtime: true }).then(() => 1)");
    const segs = await evaluate(`JSON.stringify(window.itantra.segs.slice(${n0}).map((s) => [s.language, s.timeToText, s.timings.total]))`);
    const med = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : null; };
    const rows = JSON.parse(segs), ind = rows.filter((r) => r[0] !== "en"), en = rows.filter((r) => r[0] === "en");
    const f = (v) => (v == null ? "–" : `${(v / 1000).toFixed(2)} s`);
    console.log(`[realtime] ${rows.length} parts; time-to-text Indian→target median ${f(med(ind.map((r) => r[1])))} max ${f(Math.max(...ind.map((r) => r[1])))}` +
      ` · English median ${f(med(en.map((r) => r[1])))} max ${f(Math.max(...en.map((r) => r[1])))}`);
    // peak resident memory of the tab's renderer process (page + worker + WebAssembly threads)
    const procs = fs.readdirSync("/proc").filter((p) => /^\d+$/.test(p)).map((p) => {
      try { return { cmd: fs.readFileSync(`/proc/${p}/cmdline`, "utf8"), st: fs.readFileSync(`/proc/${p}/status`, "utf8") }; } catch { return null; }
    }).filter((p) => p && p.cmd.includes(profile) && p.cmd.includes("--type=renderer"));
    const kb = (st, k) => Number((new RegExp(`${k}:\\s+(\\d+)`).exec(st) || [0, 0])[1]);
    const peak = Math.max(...procs.map((p) => kb(p.st, "VmHWM"))), now = Math.max(...procs.map((p) => kb(p.st, "VmRSS")));
    console.log(`[memory] tab renderer peak RSS ${(peak / 1024).toFixed(0)} MB, now ${(now / 1024).toFixed(0)} MB`);
    await shot("realtime");
    ws.close();
    chrome.kill();
    process.exit(0);
  }
  if (process.env.MIC) {
    await evaluate("document.querySelector('#mic').click()");
    await sleep(Number(process.env.MIC_SECONDS || 20) * 1000);
    await evaluate("document.querySelector('#mic').click()");
  } else if (process.env.OFFLINE) {
    await evaluate("window.__s = window.itantra.compose(window.itantra.cfg().languages)");
    await evaluate("window.__s.then(() => 1)");
    await cmd("Network.enable");
    await cmd("Network.emulateNetworkConditions", { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
    console.log("network: OFFLINE, navigator.onLine =", await evaluate("navigator.onLine"));
    await evaluate("window.__s.then((s) => { window.itantra.run('try', window.itantra.cfg(), s.audio, s); return 1; })");
  } else await evaluate("document.querySelector('#try-sample').click()");
  let t = await waitFor("document.querySelector('#vadstate').textContent === 'done'", 900, "try");
  console.log(`\n[Try it] ${t.toFixed(1)} s wall`);
  console.log(await evaluate("[...document.querySelectorAll('.seg')].map(s => s.querySelector('.meta').innerText.replace(/\\n/g,' ') + '\\n   ' + s.querySelector('.text').innerText + (s.querySelector('.tr') ? '\\n   → ' + s.querySelector('.tr').innerText : '') + (s.querySelector('.ref') ? '\\n   ' + s.querySelector('.ref').innerText : '')).join('\\n')"));
  await shot("try");

  if (!process.env.OFFLINE) {
    // Language test: all 10 languages as candidates
    await evaluate("document.querySelector('[data-tab=lid]').click(); document.querySelector('#lid-all').click()");
    t = await waitFor("document.querySelector('#lid-status').textContent === 'done'", 900, "language test");
    console.log(`\n[Language test] ${t.toFixed(1)} s wall:`, await evaluate("[...document.querySelectorAll('#lid-results .summary > div')].map(d => d.innerText.replace('\\n', ' ')).join(' | ')"));
    console.log(await evaluate("[...document.querySelectorAll('.lidtable tbody tr')].map(r => [...r.cells].slice(0, 8).map(c => c.innerText).join(' · ')).join('\\n')"));
    await shot("lid");
    await evaluate("document.querySelector('[data-tab=how]').click()");
    await sleep(500);
    await shot("how");
  }
  console.log("\nKPIs:", await evaluate("[...document.querySelectorAll('.kpi')].map(k => k.querySelector('.l').innerText + ': ' + k.querySelector('.v').innerText).join(' | ')"));
  console.log(await evaluate("[...document.querySelectorAll('#models tbody tr')].map(r => r.innerText.replace(/\\t/g,'  ')).join('\\n')"));
  console.log(await evaluate("[...document.querySelectorAll('#facts dt')].map(d => d.innerText + ': ' + d.nextElementSibling.innerText).join('\\n')"));
} finally {
  ws.close();
  chrome.kill();
}
