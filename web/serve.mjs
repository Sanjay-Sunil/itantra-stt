// Local static server for the browser demo. Sends the cross-origin isolation headers that multi-threaded
// WebAssembly needs (SharedArrayBuffer) and serves the models from ../models under /models/.
//   node serve.mjs [port]      then open http://localhost:8080
import fs from "fs";
import http from "http";
import path from "path";

const WEB = import.meta.dirname;
const MODELS = process.env.INDICSTT_MODELS_DIR || path.resolve(WEB, "../models");
const PORT = Number(process.argv[2] || process.env.PORT || 8080);
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".css": "text/css", ".wav": "audio/wav",
  ".svg": "image/svg+xml", ".onnx": "application/octet-stream", ".data": "application/octet-stream" };

http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const [root, rel] = url.startsWith("/models/") ? [MODELS, url.slice(8)] : [WEB, url === "/" ? "index.html" : url.slice(1)];
  const file = path.resolve(root, rel);
  if (!file.startsWith(root + path.sep) || rel.split("/").some((s) => s === "node_modules" || s.startsWith("."))) {
    res.writeHead(403).end();
    return;
  }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404).end("not found"); return; }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(file)] || "application/octet-stream",
      "Content-Length": st.size,
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": url.startsWith("/models/") ? "public, max-age=86400" : "no-cache",
    });
    fs.createReadStream(file).pipe(res);
  });
}).listen(PORT, "127.0.0.1", () => console.log(`iTantra browser demo: http://localhost:${PORT}`));
