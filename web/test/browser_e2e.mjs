// End-to-end test in a real (headless) Chromium: load the page, load models, run a sample clip, read results.
//   node serve.mjs &   node test/browser_e2e.mjs [url] [sample-button-index]
// Uses the DevTools protocol directly (Node's built-in WebSocket), so no extra dependencies.
import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

const URL_ = process.argv[2] || "http://127.0.0.1:8080/?langs=en,hi,ta";
const SAMPLE = Number(process.argv[3] || 0);
const CHROME = process.env.CHROME || "/usr/bin/chromium";
const profile = process.env.PROFILE || fs.mkdtempSync(path.join(os.tmpdir(), "itantra-e2e-")); // reuse to test the model cache
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${profile}`,
  "--no-first-run", "--autoplay-policy=no-user-gesture-required", "about:blank"], { stdio: "ignore" });
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
  const loadS = await waitFor("document.querySelector('#loadtext').textContent.startsWith('ready') || document.querySelector('#loadtext').textContent.startsWith('error')", 900, "models");
  console.log(`models: ${await evaluate("document.querySelector('#loadtext').textContent")} (wall ${loadS.toFixed(1)} s)`);
  await sleep(3000);
  console.log("storage:", await evaluate(`(async () => { const e = await navigator.storage.estimate(); const c = await caches.open("itantra-models-v1");
    const keys = await c.keys(); return JSON.stringify({ usageMB: Math.round(e.usage / 1e6), quotaMB: Math.round(e.quota / 1e6), cachedFiles: keys.length,
    encoder: keys.some((k) => k.url.endsWith("encoder.onnx.data")) }); })()`));
  await evaluate(`document.querySelectorAll('.sample')[${SAMPLE}].click()`);
  const runS = await waitFor("document.querySelector('#vadstate').textContent === 'done'", 900, "transcription");
  console.log(`transcribed in ${runS.toFixed(1)} s wall`);
  console.log(await evaluate("[...document.querySelectorAll('.seg')].reverse().map(s => s.querySelector('.meta').innerText.replace(/\\n/g,' ') + '\\n   ' + s.querySelector('.text').innerText + (s.querySelector('.tr') ? '\\n   → ' + s.querySelector('.tr').innerText : '')).join('\\n')"));
  console.log("\nKPIs:", await evaluate("[...document.querySelectorAll('.kpi')].map(k => k.querySelector('.l').innerText + ': ' + k.querySelector('.v').innerText).join(' | ')"));
  console.log(await evaluate("[...document.querySelectorAll('#models tbody tr')].map(r => r.innerText.replace(/\\t/g,'  ')).join('\\n')"));
  console.log(await evaluate("[...document.querySelectorAll('#facts dt')].map(d => d.innerText + ': ' + d.nextElementSibling.innerText).join('\\n')"));
  const shot = await cmd("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const out = process.env.SHOT || path.join(os.tmpdir(), "itantra-e2e.png");
  fs.writeFileSync(out, Buffer.from(shot.result.data, "base64"));
  console.log("screenshot:", out);
} finally {
  ws.close();
  chrome.kill();
}
