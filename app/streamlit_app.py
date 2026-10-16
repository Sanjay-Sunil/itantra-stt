"""iTantra: offline speech-to-text dashboard (public demo).

Run on the edge device:  .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
All inference runs locally on this device; visitors' browsers only record audio and display results.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
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
    eng = Engine(EngineConfig(languages=CODES, target_language="en", asr_tier=tier))
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


# ---------------------------------------------------------------- header
st.markdown('<div class="brand">iTantra</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Speak in your language.<br>Read it in theirs.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Ten Indian languages, mixed freely in one conversation. '
            '<b>Runs entirely on this device</b>, with no internet and no cloud.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- controls
c1, c2, c3 = st.columns([5, 3, 1], vertical_alignment="bottom")
langs = c1.multiselect("I speak", CODES, default=["hi", "en", "ta"], format_func=label,
                       placeholder="Choose languages")
target_opt = c2.selectbox("Show me in", CODES, index=CODES.index("en"), format_func=label)
with c3.popover("", icon=":material/tune:", help="Settings"):
    tier = st.radio("Model size", ["int8", "int4mixed"],
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
tab_rec, tab_up, tab_ex, tab_live = st.tabs(["Record", "Upload", "Examples", "Live mic"])

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
