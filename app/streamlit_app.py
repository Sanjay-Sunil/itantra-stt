"""iTantra: offline speech-to-text dashboard (public demo).

Run on the edge device:  .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
All inference runs locally on this device; visitors' browsers only record audio and display results.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from indicstt.audio import BufferSource, MicSource, load_audio
from indicstt.engine import Engine, EngineConfig
from indicstt.segment import LANGUAGES, Segment

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CODES = list(LANGUAGES)
NATIVE = {"hi": "हिन्दी", "gu": "ગુજરાતી", "mr": "मराठी", "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "ta": "தமிழ்",
          "te": "తెలుగు", "or": "ଓଡ଼ିଆ", "bn": "বাংলা", "en": "English"}
MAX_SECONDS = 120  # per recording / upload, public safeguard
MAX_CONCURRENT_JOBS = 2
CLOUD = os.environ.get("ITANTRA_CLOUD") == "1"  # hosted (e.g. Hugging Face Space): no server microphone
TIERS = [t for t in ("int8", "int4mixed") if (MODELS / f"indic-conformer-{t}").exists()] or ["int8"]

st.set_page_config(page_title="iTantra · Speech to text", page_icon="◐", layout="centered",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- style
st.markdown("""
<style>
:root { --accent:#4338CA; --ink:#18181B; --muted:#71717A; --line:#E4E4E7; --card:#FFFFFF; }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
.block-container { max-width: 760px; padding-top: 3.2rem; padding-bottom: 4rem; }
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Noto Sans",
  "Noto Sans Devanagari", "Noto Sans Tamil", Roboto, sans-serif; }
.brand { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
  font-weight: 600; margin-bottom: .4rem; }
.hero { font-size: 2.05rem; line-height: 1.15; font-weight: 650; color: var(--ink); letter-spacing: -.02em;
  margin: 0 0 .55rem 0; }
.sub { color: var(--muted); font-size: 1rem; margin-bottom: 1.8rem; }
.sub b { color: var(--ink); font-weight: 550; }
.section { font-size: .76rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin: 2.2rem 0 .7rem; }
.seg { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem;
  margin-bottom: .65rem; }
.seg-head { display:flex; align-items:center; gap:.5rem; margin-bottom:.45rem; color: var(--muted);
  font-size:.8rem; }
.chip { display:inline-flex; gap:.35rem; align-items:center; padding:.12rem .55rem; border-radius:999px;
  background:#EEF2FF; color: var(--accent); font-weight:600; font-size:.76rem; }
.chip.same { background:#F4F4F5; color:#52525B; }
.seg-time { margin-left:auto; font-variant-numeric: tabular-nums; }
.seg-main { font-size: 1.12rem; line-height: 1.5; color: var(--ink); }
.seg-orig { font-size: .9rem; line-height: 1.5; color: var(--muted); margin-top: .35rem; }
.seg-warn { font-size:.85rem; color:#B45309; }
.stats { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
.stats b { color: var(--ink); font-weight: 600; }
.empty { border: 1px dashed var(--line); border-radius: 14px; padding: 1.6rem; text-align:center;
  color: var(--muted); font-size: .92rem; }
.foot { color: var(--muted); font-size: .78rem; text-align:center; margin-top: 3rem; }
div[data-testid="stTabs"] button p { font-size: .92rem; }
.flow { display:flex; flex-direction:column; gap:0; margin: .4rem 0 1.2rem; }
.step { display:flex; gap:.8rem; }
.step-row { display:flex; gap:.6rem; flex:1; flex-wrap: wrap; }
.node { flex:1; min-width: 210px; background: var(--card); border:1px solid var(--line); border-radius: 12px;
  padding: .75rem .9rem; }
.node.accent { border-color:#C7D2FE; background:#FAFAFF; }
.node-title { font-weight: 650; color: var(--ink); font-size: .95rem; }
.node-model { color: var(--muted); font-size: .8rem; margin: .1rem 0 .45rem; }
.node-desc { color: #3F3F46; font-size: .85rem; line-height: 1.45; }
.node-metrics { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.55rem; }
.metric { font-size:.75rem; font-variant-numeric: tabular-nums; padding:.12rem .5rem; border-radius:999px;
  background:#F4F4F5; color:#3F3F46; }
.metric.t { background:#EEF2FF; color: var(--accent); }
.idx { width: 1.6rem; min-width:1.6rem; display:flex; flex-direction:column; align-items:center; }
.idx span { width:1.6rem; height:1.6rem; border-radius:999px; background: var(--accent); color:#fff;
  font-size:.75rem; font-weight:600; display:flex; align-items:center; justify-content:center; margin-top:.75rem; }
.idx i { flex:1; width:2px; background: var(--line); margin-top:.3rem; }
.par-label { font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color: var(--accent);
  font-weight:600; margin: .2rem 0 .35rem; width:100%; }
.gap { height:.6rem; }
.summary { display:flex; gap:.6rem; flex-wrap:wrap; margin:.2rem 0 1.4rem; }
.summary div { flex:1; min-width: 150px; background: var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.7rem .9rem; }
.summary b { display:block; font-size:1.35rem; color: var(--ink); font-weight:650; }
.summary small { color: var(--muted); font-size:.78rem; }
[data-testid="stMultiSelectTagsContainer"] span[data-tag] { background:#E0E7FF !important; color: var(--accent) !important;
  border-radius: 999px !important; font-weight: 550; }
[data-testid="stMultiSelectTagsContainer"] span[data-tag] * { color: var(--accent) !important; fill: var(--accent) !important; }
@media (max-width: 640px) {
  .hero { font-size: 1.7rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- shared resources (one per server)
@st.cache_resource(show_spinner=False, max_entries=2)
def base_engine(tier: str) -> Engine:
    """All languages loaded once and shared by every visitor; per-visitor options via derive()."""
    eng = Engine(EngineConfig(languages=CODES, target_language="en", asr_tier=tier, models_dir=MODELS))
    mt = eng.stages.mt
    for name in ("mt-indic-en", "mt-en-indic", "mt-indic-indic"):  # preload so no visitor waits on a load
        if (MODELS / name).exists():
            mt.bundle(name)
    return eng


@st.cache_resource(show_spinner=False)
def job_slots() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def label(code: str) -> str:
    return LANGUAGES[code] if code == "en" else f"{LANGUAGES[code]} · {NATIVE[code]}"


def mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------- rendering
def card(seg: Segment, target: str | None) -> str:
    lang = seg.language or "?"
    translated = seg.translation is not None
    chip_cls = "chip" if translated else "chip same"
    native = NATIVE.get(lang, lang)
    name = LANGUAGES.get(lang, lang)
    chip = f'<span class="{chip_cls}">{html.escape(name)}' + (
        f' <span style="font-weight:500">{html.escape(native)}</span>' if lang != "en" else "") + "</span>"
    arrow = f'<span>→ {html.escape(LANGUAGES[target])}</span>' if translated and target else ""
    head = (f'<div class="seg-head">{chip}{arrow}'
            f'<span class="seg-time">{mmss(seg.start_ms)} – {mmss(seg.end_ms)}</span></div>')
    if "asr_unavailable" in seg.flags:
        body = '<div class="seg-warn">No speech model installed for this language.</div>'
    elif "timeout" in seg.flags or "error" in seg.flags:
        body = '<div class="seg-warn">This part could not be processed.</div>'
    elif translated:
        body = (f'<div class="seg-main">{html.escape(seg.translation)}</div>'
                f'<div class="seg-orig">{html.escape(seg.text)}</div>')
    else:
        body = f'<div class="seg-main">{html.escape(seg.text) or "…"}</div>'
    return f'<div class="seg">{head}{body}</div>'


def render_cards(container, segs: list[Segment], target: str | None) -> None:
    container.markdown("".join(card(s, target) for s in segs), unsafe_allow_html=True)


def run_job(engine: Engine, source, target: str | None, audio_s: float | None, label_: str) -> None:
    slots = job_slots()
    live = st.empty()
    if not slots.acquire(blocking=False):
        live.info("Another transcription is running on this device. You're next…")
        slots.acquire()
        live.empty()
    segs: list[Segment] = []
    st.session_state["result"] = {"segments": segs, "audio_s": audio_s, "wall_s": None, "label": label_,
                                  "target": target}
    t0 = time.perf_counter()
    try:
        with st.spinner("Listening and transcribing on this device…"):
            for seg in engine.stream(source):
                fin = seg.timings.pop("finalized_at", None)
                if fin:
                    seg.timings["time_to_text"] = round((time.time() - fin) * 1000)
                segs.append(seg)
                render_cards(live, segs, target)
    finally:
        slots.release()
    st.session_state["result"]["wall_s"] = time.perf_counter() - t0
    st.rerun()


def stop_live() -> None:
    src = st.session_state.pop("live_source", None)
    if src is not None:
        src.stop()



# ---------------------------------------------------------------- "Language test" (VAD + LID)
LANG_COLOR = {"hi": "#6366F1", "gu": "#0EA5E9", "mr": "#8B5CF6", "kn": "#D97706", "ml": "#059669",
              "ta": "#DC2626", "te": "#DB2777", "or": "#0D9488", "bn": "#EA580C", "en": "#475569"}


def _truth_for(seg: Segment, refs: list[dict] | None) -> str | None:
    if not refs:
        return None
    mid = (seg.start_ms + seg.end_ms) / 2000
    for r in refs:
        if r["start_s"] - 0.3 <= mid <= r["end_s"] + 0.6:
            return r["language"]
    return None


def _timeline(rows: list[tuple[str, list[tuple[float, float, str]]]], total: float) -> str:
    out = '<div style="margin:.4rem 0 1rem">'
    for name, blocks in rows:
        out += (f'<div style="display:flex;align-items:center;gap:.6rem;margin:.3rem 0">'
                f'<div style="width:4.6rem;font-size:.75rem;color:var(--muted)">{name}</div>'
                f'<div style="position:relative;flex:1;height:2rem;background:#F4F4F5;border-radius:8px">')
        for a, b, lang in blocks:
            left, width = 100 * a / total, max(0.6, 100 * (b - a) / total)
            c = LANG_COLOR.get(lang, "#71717A")
            out += (f'<div title="{LANGUAGES.get(lang, lang)} {a:.1f}–{b:.1f}s" style="position:absolute;left:{left:.2f}%;'
                    f'width:{width:.2f}%;top:3px;bottom:3px;border-radius:6px;background:{c}22;'
                    f'border:1px solid {c}66;color:{c};font-size:.7rem;font-weight:650;display:flex;'
                    f'align-items:center;justify-content:center;overflow:hidden">{lang}</div>')
        out += "</div></div>"
    ticks = "".join(f'<span>{t}s</span>' for t in range(0, int(total) + 1, max(5, int(total) // 8 or 1)))
    out += (f'<div style="display:flex;justify-content:space-between;margin-left:5.2rem;font-size:.68rem;'
            f'color:var(--muted)">{ticks}</div></div>')
    return out


def render_lid_test() -> None:
    st.markdown('<div class="sub" style="margin-bottom:1rem">Speak in any of the 10 languages, switching after '
                'a short pause. Voice detection cuts the audio into chunks at pauses and each chunk is tagged '
                'with its language. All 10 languages are candidates here.</div>', unsafe_allow_html=True)
    demo = ROOT / "data/demo/all10_languages.wav"
    src = st.segmented_control("Audio", ["Example: all 10 languages", "Record", "Upload"],
                               default="Example: all 10 languages", label_visibility="collapsed", key="lid_src")
    audio, refs, label_ = None, None, None
    if src == "Record":
        rec = st.audio_input("Record", label_visibility="collapsed", key="lid_rec")
        if rec is not None:
            audio, label_ = load_audio(io.BytesIO(rec.getvalue())), "Your recording"
    elif src == "Upload":
        up = st.file_uploader("Audio", type=["wav", "flac", "mp3", "ogg"], label_visibility="collapsed",
                              key="lid_up")
        if up is not None:
            audio, label_ = load_audio(up), up.name
    elif demo.exists():
        audio, label_ = load_audio(demo), "all 10 languages"
        meta = demo.with_suffix(".json")
        refs = json.loads(meta.read_text()) if meta.exists() else None
        st.audio(str(demo))
        st.caption("7 s of read speech per language (FLEURS, CC-BY-4.0): " +
                   " → ".join(LANGUAGES[r["language"]] for r in refs or []))
    if audio is not None and st.button("Detect languages", type="primary", key="lid_go"):
        if len(audio) / 16000 > MAX_SECONDS:
            st.warning(f"Audio is limited to {MAX_SECONDS // 60} minutes.")
        else:
            eng = base_engine("int8").derive(languages=CODES, target_language=None, asr_mode="ctc")
            slots = job_slots()
            slots.acquire()
            try:
                with st.spinner("Cutting into chunks and identifying languages…"):
                    segs = list(eng.stream(BufferSource(audio)))
            finally:
                slots.release()
            st.session_state["lid_test"] = {"segments": segs, "total": len(audio) / 16000, "refs": refs,
                                            "label": label_}
    res = st.session_state.get("lid_test")
    if not res:
        st.markdown('<div class="empty">Chunks and their languages will appear here.</div>', unsafe_allow_html=True)
        return
    segs, refs, total = res["segments"], res["refs"], res["total"]
    truth = [_truth_for(s, refs) for s in segs]
    judged = [(s.language == t) for s, t in zip(segs, truth, strict=True) if t]
    tiles = (f'<div><b>{len(segs)}</b><small>chunks found by voice detection</small></div>'
             f'<div><b>{len({s.language for s in segs})}</b><small>languages detected</small></div>')
    if judged:
        tiles += f'<div><b>{sum(judged)}/{len(judged)}</b><small>chunks tagged correctly</small></div>'
    lid_ms = [s.timings.get("lid", 0) + s.timings.get("lid_ctc", 0) for s in segs]
    tiles += f'<div><b>{_median(lid_ms):.0f} ms</b><small>language ID per chunk (median)</small></div>'
    st.markdown(f'<div class="summary">{tiles}</div>', unsafe_allow_html=True)
    rows = [("Detected", [(s.start_ms / 1000, s.end_ms / 1000, s.language) for s in segs])]
    if refs:
        rows.append(("Actual", [(r["start_s"], r["end_s"], r["language"]) for r in refs]))
    st.markdown(_timeline(rows, total), unsafe_allow_html=True)
    table = []
    for s, t in zip(segs, truth, strict=True):
        d = getattr(s, "lid_detail", None) or {}
        top = ", ".join(f"{k} {v:.2f}" for k, v in d.get("lid_top", []))
        table.append({"#": s.seq, "time": f"{mmss(s.start_ms)}–{mmss(s.end_ms)}",
                      "detected": label(s.language) if s.language in LANGUAGES else s.language,
                      "confidence": round(s.lid_confidence, 2),
                      "decided by": {"lid": "AmberNet", "ctc": "speech-model check", "lid+ctc": "AmberNet, confirmed",
                                     "manual": "manual",
                                     "single": "only option"}.get(d.get("decided_by"), "—"),
                      "AmberNet top 3": top or "—",
                      **({"actual": label(t) if t else "—", "ok": "✓" if t and t == s.language else ("✗" if t else "")}
                         if refs else {}),
                      "text (start)": (s.text or "")[:40]})
    st.dataframe(pd.DataFrame(table).set_index("#"), width="stretch")
    st.caption("AmberNet has no Odia class: when Odia is possible it only decides English vs Indian, and the "
               "speech model's CTC check picks the Indian language. Close pairs (e.g. Hindi/Marathi) are "
               "double-checked the same way when AmberNet is unsure.")


# ---------------------------------------------------------------- "How does this work?"
@st.cache_data(show_spinner=False)
def stage_profile(tier: str) -> dict:
    p = ROOT / f"results/stage-profile-{tier}.json"
    if not p.exists():
        p = ROOT / "results/stage-profile-int8.json"
    return json.loads(p.read_text()) if p.exists() else {"memory": [], "latency_median_ms": {}}


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    return vals[len(vals) // 2] if vals else None


def render_how(res: dict | None) -> None:
    prof = stage_profile("int8")
    mem = {r["component"]: r for r in prof["memory"]}
    lat = dict(prof["latency_median_ms"])
    source = "reference run (63 s English–Hindi–Tamil clip, real-time)"
    if res and res.get("segments"):  # prefer the visitor's own last run
        segs = res["segments"]
        live = {}
        for key in ("lid", "encoder", "decode", "mt", "wait_reorder", "time_to_text", "vad"):
            for grp, sel in (("indic", lambda s: s.language != "en"), ("en", lambda s: s.language == "en")):
                v = _median([s.timings.get(key) for s in segs if sel(s)])
                if v is not None:
                    live[f"{key}:{grp}"] = v
        if live:
            lat.update(live)
            source = "your last run"

    def ms(key):
        v = lat.get(key)
        return "—" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")

    def ram(*names):
        rows = [mem[n] for n in names if n in mem]
        if not rows:
            return "—"
        priv = sum(r["private_mb"] for r in rows)
        fmap = sum(r["file_mb"] for r in rows)
        return f"{priv} MB" + (f" + {fmap} MB mapped" if fmap > 20 else "")

    def node(title, model, desc, t=None, r=None, accent=False):
        metrics = ""
        if t:
            metrics += f'<span class="metric t">⏱ {t}</span>'
        if r:
            metrics += f'<span class="metric">RAM {r}</span>'
        return (f'<div class="node{" accent" if accent else ""}"><div class="node-title">{title}</div>'
                f'<div class="node-model">{model}</div><div class="node-desc">{desc}</div>'
                f'<div class="node-metrics">{metrics}</div></div>')

    def step(i, inner, last=False):
        line = "" if last else "<i></i>"
        return f'<div class="step"><div class="idx"><span>{i}</span>{line}</div><div class="step-row">{inner}</div></div><div class="gap"></div>'

    total_priv = sum(r["private_mb"] for r in mem.values())
    total_file = sum(r["file_mb"] for r in mem.values())
    st.markdown(
        '<div class="sub" style="margin-bottom:1rem">Every step runs on this device. Speech is cut into parts '
        'at natural pauses; each part flows through the steps below while the next part is still being spoken. '
        f'Timings are medians per part from the <b>{source}</b>.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary">'
        f'<div><b>{ms("time_to_text:indic")}</b><small>speech ends → translated text (Indian languages)</small></div>'
        f'<div><b>{ms("time_to_text:en")}</b><small>speech ends → text (English)</small></div>'
        f'<div><b>{(total_priv + total_file) / 1024:.1f} GB</b><small>RAM, all models loaded '
        f'({total_priv} MB private + {total_file} MB memory-mapped)</small></div></div>',
        unsafe_allow_html=True)

    flow = '<div class="flow">'
    flow += step(1, node("Audio in", "Microphone or file · 16 kHz mono",
                         "Sound arrives as a stream of 32 ms chunks (512 samples).", None, ram("runtime")))
    flow += step(2, node("Voice detection", "Silero VAD · 2 MB",
                         "Scores every chunk for speech. After 0.4 s of silence it closes a numbered part "
                         "(seq). Its compute is spread across the speech, so it adds only the pause wait.",
                         "0.4 s pause wait", ram("vad")))
    par = ('<div class="par-label">In parallel on each part</div>'
           + node("Language ID", "AmberNet (TitaNet family) · INT8",
                  "Which of your languages is this? Odia and close pairs are double-checked by the speech model.",
                  ms("lid:indic"), ram("lid"), accent=True)
           + node("Speech encoder", "IndicConformer 600M · INT8",
                  "Turns sound into acoustic features shared by all 9 Indian languages, so it starts before the "
                  "language is known.", ms("encoder:indic"), ram("encoder"), accent=True))
    flow += step(3, par)
    flow += step(4, node("Decoder", "RNN-T head for the detected language · INT8",
                         "Writes the text using that language's vocabulary. English parts go to Whisper tiny "
                         f"instead ({ms('decode:en')} decode).", ms("decode:indic"),
                         "in encoder · Whisper " + ram("english")))
    flow += step(5, node("Translation", "IndicTrans2 distilled · INT8",
                         "Only when the part's language differs from yours. Parts already in your language skip it.",
                         ms("mt:indic"), ram("mt")))
    flow += step(6, node("Ordering", "Reorder buffer",
                         "Parts finish at different speeds; part N is shown only after part N−1, so the transcript "
                         "reads in speaking order.", ms("wait_reorder:indic") + " wait", None), last=True)
    flow += "</div>"
    st.markdown(flow, unsafe_allow_html=True)

    st.markdown('<div class="section">How does this work?</div>', unsafe_allow_html=True)
    table = pd.DataFrame([
        ["1", "Audio in", "Stream cut into 32 ms chunks", "—", "streaming", ram("runtime")],
        ["2", "Voice detection", "Finds speech, closes a part after 0.4 s pause", "Silero VAD", "0.4 s pause wait",
         ram("vad")],
        ["3a", "Language ID", "Detects the language (parallel with 3b)", "AmberNet · INT8", ms("lid:indic"),
         ram("lid")],
        ["3b", "Speech encoder", "Acoustic features, shared by 9 languages", "IndicConformer 600M · INT8",
         ms("encoder:indic"), ram("encoder")],
        ["4", "Decoder", "Text in the detected language", "RNN-T head · INT8", ms("decode:indic"), "in 3b"],
        ["4′", "English decoder", "Text for English parts", "Whisper tiny.en · INT8",
         ms("decode:en"), ram("english")],
        ["5", "Translation", "Only if language ≠ yours", "IndicTrans2 · INT8", ms("mt:indic"), ram("mt")],
        ["6", "Ordering", "Releases parts in speaking order", "Reorder buffer", ms("wait_reorder:indic"), "—"],
        ["", "Total", "Speech ends → text on screen", "", ms("time_to_text:indic"),
         f"{total_priv} MB + {total_file} MB mapped"],
    ], columns=["Step", "Stage", "What happens", "Model", "Latency (median / part)", "RAM"])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("RAM: private = memory only this app uses; mapped = model files paged in from disk, which the "
               "system can reclaim and share. Measured on an i7 laptop, CPU only "
               "(`scripts/measure_memory.py`). Translation RAM is for one direction; each extra direction "
               "adds about the same.")


# ---------------------------------------------------------------- header
st.markdown('<div class="brand">iTantra</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Speak in your language.<br>Read it in theirs.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Ten Indian languages, mixed freely in one conversation. '
            '<b>Runs entirely on this device</b>, with no internet and no cloud.</div>', unsafe_allow_html=True)

page_try, page_lid, page_how = st.tabs(["Try it", "Language test", "How does this work?"])

with page_how:
    render_how(st.session_state.get("result"))

with page_lid:
    render_lid_test()

with page_try:
    # ---------------------------------------------------------------- controls
    c1, c2, c3 = st.columns([5, 3, 1], vertical_alignment="bottom")
    langs = c1.multiselect("I speak", CODES, default=["hi", "en", "ta"], format_func=label,
                           placeholder="Choose languages")
    target_opt = c2.selectbox("Show me in", CODES, index=CODES.index("en"), format_func=label)
    with c3.popover("", icon=":material/tune:", help="Settings"):
        tier = st.radio("Model size", TIERS,
                        format_func={"int8": "Standard (INT8)", "int4mixed": "Compact (INT4)"}.get)
        mode = st.radio("Decoder", ["rnnt", "ctc"], format_func={"rnnt": "Accurate (RNN-T)", "ctc": "Fast (CTC)"}.get)
        auto = st.toggle("Detect language automatically", value=True)
        manual = None if auto else st.selectbox("Spoken language", langs or CODES, format_func=label)

    if not langs:
        st.markdown('<div class="empty">Choose at least one language you speak.</div>', unsafe_allow_html=True)
        st.stop()

    with st.spinner("Starting the speech engine… (first visit only)"):
        base = base_engine(tier)
    engine = base.derive(languages=langs, target_language=target_opt, manual_language=manual, asr_mode=mode)
    target = target_opt

    # ---------------------------------------------------------------- input
    tab_names = ["Record", "Upload", "Examples"] + ([] if CLOUD else ["Live mic"])
    tabs = st.tabs(tab_names)
    tab_rec, tab_up, tab_ex = tabs[:3]
    tab_live = tabs[3] if not CLOUD else None

    with tab_rec:
        rec = st.audio_input("Record", label_visibility="collapsed")
        st.caption(f"Tap the mic, speak (up to {MAX_SECONDS // 60} min), tap again to stop. "
                   "Transcription starts automatically.")
        if rec is not None:
            data = rec.getvalue()
            key = hashlib.sha1(data).hexdigest()
            if st.session_state.get("last_rec") != key:
                st.session_state["last_rec"] = key
                audio = load_audio(io.BytesIO(data))
                if len(audio) / 16000 > MAX_SECONDS:
                    st.warning(f"Recordings are limited to {MAX_SECONDS // 60} minutes.")
                else:
                    run_job(engine, BufferSource(audio), target, len(audio) / 16000, "Your recording")

    with tab_up:
        up = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "ogg"], label_visibility="collapsed")
        if up is not None and st.button("Transcribe", type="primary", key="go_up"):
            audio = load_audio(up)
            if len(audio) / 16000 > MAX_SECONDS:
                st.warning(f"Files are limited to {MAX_SECONDS // 60} minutes.")
            else:
                run_job(engine, BufferSource(audio), target, len(audio) / 16000, up.name)

    with tab_ex:
        examples = {}
        for p in sorted((ROOT / "data/demo").glob("*.wav")):
            meta = p.with_suffix(".json")
            parts = [r["language"] for r in json.loads(meta.read_text())] if meta.exists() else []
            examples[" → ".join(LANGUAGES[x] for x in dict.fromkeys(parts)) + " (mixed)" if parts else p.stem] = p
        for code in CODES:
            first = next(iter(sorted((ROOT / "data/smoke" / code).glob("*.wav"))), None)
            if first:
                examples[f"{label(code)}"] = first
        if examples:
            name = st.selectbox("Example", list(examples), label_visibility="collapsed")
            path = examples[name]
            st.audio(str(path))
            st.caption("Read speech from the FLEURS dataset (CC-BY-4.0). Include its languages in “I speak”.")
            if st.button("Transcribe example", type="primary", key="go_ex"):
                audio = load_audio(path)
                run_job(engine, BufferSource(audio), target, len(audio) / 16000, name)
        else:
            st.markdown('<div class="empty">No examples on this device.</div>', unsafe_allow_html=True)

    if tab_live is not None:  # the server's own microphone: only meaningful on a device you stand at
        with tab_live:
            st.caption("Uses the microphone of the device running iTantra. Text appears after each pause.")
            a, b = st.columns(2)
            start = a.button("Start listening", type="primary", width="stretch")
            if b.button("Stop", width="stretch"):
                stop_live()
            if start:
                stop_live()
                src = MicSource()
                st.session_state["live_source"] = src
                run_job(engine, src, target, None, "Live microphone")

    # ---------------------------------------------------------------- result
    res = st.session_state.get("result")
    st.markdown('<div class="section">Transcript</div>', unsafe_allow_html=True)
    if not res or not res["segments"]:
        st.markdown('<div class="empty">Your words will appear here, each part in the language you chose.</div>',
                    unsafe_allow_html=True)
    else:
        segs: list[Segment] = res["segments"]
        render_cards(st.container(), segs, res["target"])
        langs_heard = list(dict.fromkeys(LANGUAGES.get(s.language, s.language) for s in segs if s.language))
        parts = [f"<b>{len(segs)}</b> part{'s' if len(segs) != 1 else ''}",
                 "heard " + ", ".join(langs_heard)]
        if res["audio_s"] and res["wall_s"]:
            parts.append(f"<b>{res['audio_s']:.0f} s</b> of audio in <b>{res['wall_s']:.1f} s</b>")
        st.markdown(f'<div class="stats">{" · ".join(parts)} · processed on this device</div>',
                    unsafe_allow_html=True)

        text = "\n".join(s.output_text for s in segs if s.output_text)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download text", text, file_name="transcript.txt", width="stretch")
        d2.download_button("Download JSON", json.dumps([s.to_dict() for s in segs], ensure_ascii=False, indent=1),
                           file_name="segments.json", width="stretch")

        with st.expander("Under the hood"):
            st.caption("Silero VAD → AmberNet language ID ‖ IndicConformer encoder → RNN-T / CTC "
                       "(Whisper for English) → IndicTrans2 translation → reorder buffer. All models INT8 or lower.")
            rows = []
            for s in segs:
                t = s.timings
                rows.append({"#": s.seq, "language": s.language, "confidence": s.lid_confidence,
                             "audio s": round((s.end_ms - s.start_ms) / 1000, 1),
                             "lang ID ms": t.get("lid"), "speech model ms": (t.get("encoder") or 0) + (t.get("decode") or 0),
                             "translate ms": t.get("mt"), "time to text ms": t.get("time_to_text")})
            st.dataframe(pd.DataFrame(rows).set_index("#"), width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("Memory in use", f"{rss_mb() / 1024:.1f} GB")
            m2.metric("Speech model", "INT4-mixed" if tier == "int4mixed" else "INT8")
            m3.metric("Decoder", mode.upper())


st.markdown('<div class="foot">iTantra · offline speech-to-text for 10 Indian languages</div>',
            unsafe_allow_html=True)
