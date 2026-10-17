// "How it works": the pipeline and where it runs, as diagrams with live numbers from this session.

const median = (v) => { const s = v.filter((x) => x > 0).sort((a, b) => a - b); return s.length ? s[s.length >> 1] : null; };
const ms = (v) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);
const mb = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${Math.round(b / 1e6)} MB`);
const g = (macs) => (macs ? `${(macs / 1e9).toFixed(macs < 1e10 ? 1 : 0)} GMAC` : "");

function box(x, y, w, h, title, sub, live, cls = "", note = "") {
  return `<g class="d-box ${cls}" transform="translate(${x},${y})">
    <rect width="${w}" height="${h}" rx="8"/>
    <text x="${w / 2}" y="22" class="d-title">${title}</text>
    <text x="${w / 2}" y="39" class="d-sub">${sub}</text>
    ${note ? `<text x="${w / 2}" y="54" class="d-sub note">${note}</text>` : ""}
    ${live ? `<text x="${w / 2}" y="${h - 11}" class="d-live">${live}</text>` : ""}</g>`;
}
const arrow = (d, label, lx, ly, cls = "") =>
  `<path d="${d}" class="d-arrow ${cls}" marker-end="url(#ah)"/>${label ? `<text x="${lx}" y="${ly}" class="d-label">${label}</text>` : ""}`;
const defs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" class="d-head"/></marker></defs>`;

/** segs: all processed segments; rs: run stats per label; size: bytes per model dir */
export function renderHow(el, { segs, rs, size, threads, target, tier = "int8" }) {
  const ind = segs.filter((s) => s.language !== "en"), en = segs.filter((s) => s.language === "en");
  const t = (list, k) => median(list.map((s) => s.timings[k]));
  const macs = (...k) => k.reduce((a, x) => a + (rs[x]?.macs || 0), 0);
  const live = (time, m) => [time !== "–" ? time : "", g(m)].filter(Boolean).join(" · ");
  const mtSize = (size["mt-indic-en"] || 0);

  const pipeline = `<svg class="diagram" viewBox="0 0 840 300" role="img" aria-label="Pipeline diagram">${defs}
    ${box(6, 108, 104, 84, "Audio in", "mic or file", "", "muted", "16 kHz mono")}
    ${box(136, 108, 140, 84, "Voice detection", `Silero · ${mb(size["silero-vad"] || 2.2e6)}`, live(ms(t(segs, "vad")), macs("vad")), "", "cuts at 0.4 s pauses")}
    ${box(302, 108, 150, 84, "Language ID", `AmberNet · ${mb(size["lid-ambernet"] || 3e7)}`, live(ms(t(segs, "lid")), macs("lid")), "", "+ speech-model check")}
    ${box(510, 14, 176, 84, "Indian speech → text", `IndicConformer 600M ${tier === "int4mixed" ? "INT4" : "INT8"} · ${mb(size[`indic-conformer-${tier}`] || 6.7e8)}`,
      live(ms((t(ind, "encoder") || 0) + (t(ind, "decode") || 0) || null), macs("encoder", "ctc", "rnnt_pred", "rnnt_joint")), "accent", "9 languages, one encoder")}
    ${box(510, 202, 176, 84, "English speech → text", `Whisper tiny · ${mb(size["whisper-tiny-en"] || 4.4e7)}`,
      live(ms((t(en, "encoder") || 0) + (t(en, "decode") || 0) || null), macs("whisper_enc", "whisper_dec")), "", "English parts only")}
    ${box(712, 108, 124, 84, "Translation", `IndicTrans2 · ${mb(mtSize || 2.6e8)}`, live(ms(t(segs, "mt")), macs("mt")), "good", `if ≠ ${target || "target"}`)}
    ${arrow("M110,150 L134,150")}${arrow("M276,150 L300,150")}
    ${arrow("M452,136 C482,136 480,56 508,56")}${arrow("M452,164 C482,164 480,244 508,244")}
    <text x="502" y="44" class="d-label end">Indian</text><text x="502" y="276" class="d-label end">English</text>
    ${arrow("M686,56 C704,56 700,132 710,136")}${arrow("M686,244 C704,244 700,168 710,164")}
  </svg>`;

  const where = `<svg class="diagram" viewBox="0 0 780 210" role="img" aria-label="Where the models run">${defs}
    <rect x="130" y="10" width="470" height="190" rx="14" class="d-frame"/>
    <text x="146" y="32" class="d-frame-label">Your browser tab</text>
    ${box(150, 50, 130, 60, "Page", "controls · results", "", "muted")}
    ${box(320, 44, 260, 72, "Web Worker", "ONNX Runtime Web · WASM SIMD", `${threads} threads · ${g(macs(...Object.keys(rs)))}`, "accent")}
    ${box(320, 134, 230, 54, "Cache Storage", "models kept on disk", "", "muted")}
    ${box(10, 50, 96, 60, "Microphone", "or a file", "", "muted")}
    ${box(640, 128, 130, 60, "Model host", "first visit only", "", "muted")}
    ${arrow("M106,80 L148,80", "audio", 127, 70)}
    ${arrow("M280,72 L318,72", "", 0, 0)}${arrow("M318,92 L282,92", "", 0, 0)}
    ${arrow("M435,134 L435,118", "", 0, 0)}
    ${arrow("M640,161 L552,161")}
    <g class="d-no"><path d="M600,80 L740,80"/><text x="670" y="66" class="d-label">no audio to any server</text>
      <path d="M660,70 L680,90 M680,70 L660,90" class="d-x"/></g>
  </svg>`;

  const perSec = segs.length ? macs(...Object.keys(rs)) / 1e9 / (segs.reduce((a, s) => a + s.durationMs, 0) / 1000) : null;
  el.innerHTML = `
    <h3 class="d-h">The pipeline</h3>${pipeline}
    <h3 class="d-h">Where it runs</h3>${where}
    <p class="hint small">${segs.length
      ? `From your ${segs.length} parts: medians per part, compute summed. ${perSec.toFixed(1)} GMAC per second of speech.`
      : "Numbers fill in after you run audio (medians per part, compute summed)."}
      The native engine runs language ID and the encoder in parallel; here language ID runs first, so English parts skip
      the Indian-language encoder.</p>`;
}
