// Builds a sample clip for any language combination from the per-language FLEURS clips (samples/clips.json),
// with the true language, timing and text of each part, so results can be checked against it.

const SR = 16000;
let clipsP = null;
const cache = new Map();

export function loadClips() {
  clipsP ||= fetch("samples/clips.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  return clipsP;
}

export function parseWav(buf) {
  const v = new DataView(buf);
  let off = 12;
  while (off < v.byteLength) {
    const id = String.fromCharCode(v.getUint8(off), v.getUint8(off + 1), v.getUint8(off + 2), v.getUint8(off + 3));
    const size = v.getUint32(off + 4, true);
    if (id === "data") {
      const out = new Float32Array(size / 2);
      for (let i = 0; i < out.length; i++) out[i] = v.getInt16(off + 8 + 2 * i, true) / 32768;
      return out;
    }
    off += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}

async function clipAudio(file) {
  if (!cache.has(file)) cache.set(file, fetch(`samples/${file}`).then((r) => r.arrayBuffer()).then(parseWav));
  return cache.get(file);
}

export function toWavUrl(f32) {
  const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true);
  v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) v.setInt16(44 + 2 * i, Math.max(-1, Math.min(1, f32[i])) * 32767, true);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

/** Alternate the languages (one clip each per round) up to maxSeconds, with a pause between parts. */
export async function compose(langs, { maxSeconds = 50, gap = 1.0 } = {}) {
  const clips = await loadClips();
  const avail = langs.filter((l) => clips[l]?.length);
  if (!avail.length) return null;
  const order = [];
  let total = 0;
  for (let round = 0; round < 2; round++)
    for (const l of avail) {
      const c = clips[l][round % clips[l].length];
      if (round > 0 && (total + c.seconds > maxSeconds || clips[l].length <= round)) continue;
      order.push({ lang: l, ...c });
      total += c.seconds + gap;
    }
  const parts = await Promise.all(order.map((o) => clipAudio(o.file)));
  const g = Math.round(gap * SR);
  const audio = new Float32Array(g + parts.reduce((a, p) => a + p.length + g, 0));
  const refs = [];
  let pos = g;
  order.forEach((o, i) => {
    audio.set(parts[i], pos);
    refs.push({ language: o.lang, start_s: pos / SR, end_s: (pos + parts[i].length) / SR, text: o.text });
    pos += parts[i].length + g;
  });
  return { audio, refs, url: toWavUrl(audio) };
}

/** The reference part a detected segment falls in (by its midpoint), as in the dashboard's language test. */
export function refFor(seg, refs) {
  if (!refs) return null;
  const mid = (seg.startMs + seg.endMs) / 2000;
  return refs.find((r) => r.start_s - 0.3 <= mid && mid <= r.end_s + 0.6) || null;
}
