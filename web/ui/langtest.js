// Language test view: chunks from voice detection, their detected vs actual language, and the VAD trace.
import { LANGUAGES } from "../lib/engine.js";
import { refFor } from "./samples.js";

export const LANG_COLOR = { hi: "#e8590c", gu: "#2f9e44", mr: "#d6336c", kn: "#1c7ed6", ml: "#0c8599", ta: "#7048e8",
  te: "#c2255c", or: "#f08c00", bn: "#5c940d", en: "#868e96" };
const DECIDED = { lid: "AmberNet", ctc: "speech-model check", "lid+ctc": "AmberNet + check", manual: "manual", single: "only option" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const median = (v) => { const s = v.filter((x) => x != null).sort((a, b) => a - b); return s.length ? s[s.length >> 1] : null; };

function row(name, blocks, total) {
  return `<div class="tl-row"><span class="tl-name">${name}</span><div class="tl-track">${blocks.map(([a, b, l]) =>
    `<i style="left:${(100 * a) / total}%;width:${Math.max(0.6, (100 * (b - a)) / total)}%;--c:${LANG_COLOR[l] || "#868e96"}"
      title="${LANGUAGES[l] || l} ${a.toFixed(1)}–${b.toFixed(1)} s">${l}</i>`).join("")}</div></div>`;
}

function vadTrace(trace, segs, total) {
  // speech probability per 32 ms window, start/stop thresholds, and the chunks it produced
  const W = 1000, H = 90, n = trace.length || 1, x = (i) => (W * i) / n, y = (p) => H - 4 - p * (H - 8);
  const pts = Array.from(trace, (p, i) => `${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(" ");
  const spans = segs.map((s) => `<rect x="${(W * s.startMs) / 1000 / total}" y="0" width="${(W * (s.endMs - s.startMs)) / 1000 / total}"
    height="${H}" class="vad-span"/>`).join("");
  return `<div class="tl-row"><span class="tl-name">Voice</span><svg class="vad-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
    role="img" aria-label="Voice activity probability over time">${spans}
    <line x1="0" x2="${W}" y1="${y(0.5)}" y2="${y(0.5)}" class="vad-th"/><line x1="0" x2="${W}" y1="${y(0.35)}" y2="${y(0.35)}" class="vad-th low"/>
    <polyline points="${pts}" class="vad-line"/></svg></div>
    <div class="tl-legend"><span><i class="sw line"></i>speech probability</span><span><i class="sw th"></i>start 0.5 · stop 0.35</span>
    <span><i class="sw span"></i>chunk (closed after 0.4 s below stop)</span></div>`;
}

export function renderLidTest(el, { segs, refs, trace, total }) {
  const truth = segs.map((s) => refFor(s, refs)?.language || null);
  const judged = truth.map((t, i) => (t ? segs[i].language === t : null)).filter((x) => x !== null);
  const lidMs = median(segs.map((s) => (s.timings.lid || 0) + (s.timings.lid_ctc || 0)));
  const tiles = [[segs.length, "chunks found"], [new Set(segs.map((s) => s.language)).size, "languages detected"],
    ...(judged.length ? [[`${judged.filter(Boolean).length}/${judged.length}`, "tagged correctly"]] : []),
    [lidMs != null ? `${lidMs.toFixed(0)} ms` : "–", "language ID per chunk"]];
  const ticks = Array.from({ length: Math.floor(total / 10) + 1 }, (_, i) => `<span style="left:${(1000 * i) / total}%">${i * 10}s</span>`).join("");
  el.innerHTML = `
    <div class="summary">${tiles.map(([v, l]) => `<div><b>${v}</b><small>${l}</small></div>`).join("")}</div>
    <div class="timeline">
      ${row("Detected", segs.map((s) => [s.startMs / 1000, s.endMs / 1000, s.language]), total)}
      ${refs ? row("Actual", refs.map((r) => [r.start_s, r.end_s, r.language]), total) : ""}
      ${trace?.length ? vadTrace(trace, segs, total) : ""}
      <div class="tl-ticks">${ticks}</div>
    </div>
    <div class="table-wrap"><table class="lidtable">
      <thead><tr><th>#</th><th>Time</th><th>Detected</th><th>Conf.</th><th>Decided by</th><th>AmberNet top 3</th>
        ${refs ? "<th>Actual</th><th></th>" : ""}<th>Text</th></tr></thead>
      <tbody>${segs.map((s, i) => `<tr>
        <td>${s.seq}</td><td>${(s.startMs / 1000).toFixed(1)}–${(s.endMs / 1000).toFixed(1)} s</td>
        <td><span class="chip" style="--c:${LANG_COLOR[s.language]}">${LANGUAGES[s.language] || s.language}</span></td>
        <td>${(s.confidence ?? 0).toFixed(2)}</td><td>${DECIDED[s.decidedBy] || "–"}</td>
        <td class="mono">${(s.lidTop || []).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ") || "–"}</td>
        ${refs ? `<td>${truth[i] ? LANGUAGES[truth[i]] : "–"}</td><td>${truth[i] ? (truth[i] === s.language ? "✓" : "✗") : ""}</td>` : ""}
        <td class="clip" lang="${s.language}">${esc((s.text || "").slice(0, 48))}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="hint small">AmberNet has no Odia class: when Odia is possible it only decides English vs Indian, and the speech
      model's CTC scores pick the Indian language. Close pairs (Hindi/Marathi) get the same check when AmberNet is unsure.</p>`;
}
