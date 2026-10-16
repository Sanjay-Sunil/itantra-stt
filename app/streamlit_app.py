"""iTantra STT engine — edge demo UI.

Run on the edge device:  .venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
Everything runs locally and offline; the browser is only a display (and optionally a microphone).
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from indicstt.audio import BufferSource, MicSource, load_audio
from indicstt.engine import FLAG_NO_ASR, Engine, EngineConfig
from indicstt.segment import LANGUAGES, Segment

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CODES = list(LANGUAGES)
NAME = LANGUAGES

st.set_page_config(page_title="iTantra STT", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------- helpers
def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


@st.cache_resource(show_spinner=False, max_entries=2)
def get_engine(langs: tuple[str, ...], target: str | None, manual: str | None, workers: int) -> Engine:
    return Engine(EngineConfig(languages=list(langs), target_language=target,
                               manual_language=manual, asr_workers=workers))


def stop_running_source() -> None:
    src = st.session_state.pop("live_source", None)
    if src is not None:
        src.stop()


def segment_card(seg: Segment, target: str | None) -> None:
    lang = NAME.get(seg.language, seg.language or "?")
    t0, t1 = seg.start_ms / 1000, seg.end_ms / 1000
    flags = " ".join(f"`{f}`" for f in seg.flags)
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        c1.markdown(f"**#{seg.seq}** · {t0:.1f}–{t1:.1f}s")
        c1.markdown(f"🗣️ **{lang}** ({seg.lid_confidence:.2f})")
        if flags:
            c1.markdown(flags)
        if FLAG_NO_ASR in seg.flags:
            c2.warning(f"No ASR model installed for {lang} yet.")
            return
        if seg.translation is not None:
            c2.markdown(f"### {seg.translation}")
            c2.caption(f"original ({lang}): {seg.text}")
        else:
            c2.markdown(f"### {seg.text or '…'}")
            if target and seg.language == target:
                c2.caption("already in target language — MT skipped")


def timings_frame(segs: list[Segment]) -> pd.DataFrame:
    rows = []
    for s in segs:
        t = {k: v for k, v in s.timings.items() if k != "finalized_at"}
        rows.append({"seq": s.seq, "lang": s.language, "audio_s": round((s.end_ms - s.start_ms) / 1000, 2), **t})
    return pd.DataFrame(rows).set_index("seq") if rows else pd.DataFrame()


def run_stream(engine: Engine, source, target: str | None, audio_seconds: float | None = None) -> None:
    """Consume engine.stream(source), rendering each segment as soon as the reorder buffer releases it."""
    segs: list[Segment] = []
    # stored up-front so segments survive a Stop click (which interrupts this script run)
    st.session_state["last_run"] = {"segments": segs, "wall_s": 0.0, "audio_s": audio_seconds}
    live = st.empty()
    stats = st.empty()
    t0 = time.perf_counter()
    for seg in engine.stream(source):
        fin = seg.timings.get("finalized_at")
        if fin:
            seg.timings["time_to_text_ms"] = round((time.time() - fin) * 1000, 1)
        segs.append(seg)
        with live.container():
            for s in segs:
                segment_card(s, target)
        stats.caption(f"{len(segs)} segments · RSS {rss_mb():.0f} MB")
    wall = time.perf_counter() - t0
    st.session_state["last_run"] = {"segments": segs, "wall_s": wall, "audio_s": audio_seconds}
    st.rerun()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎙️ iTantra STT")
st.sidebar.caption("Offline · on-device · 10 languages")
langs = st.sidebar.multiselect("Speaker may use", CODES, default=["hi", "en", "ta"], format_func=lambda c: NAME[c])
target_opt = st.sidebar.selectbox("Deliver everything in", ["none", *CODES], index=1 + CODES.index("en"),
                                  format_func=lambda c: "No translation" if c == "none" else NAME[c])
target = None if target_opt == "none" else target_opt
lid_mode = st.sidebar.radio("Language detection", ["Auto (LID)", "Manual"], horizontal=True)
manual = None
if lid_mode == "Manual":
    manual = st.sidebar.selectbox("Spoken language", langs or CODES, format_func=lambda c: NAME[c])
workers = st.sidebar.slider("ASR workers", 1, 3, 2)

if not langs:
    st.info("Pick at least one language in the sidebar.")
    st.stop()

with st.spinner("Loading models (first time only)…"):
    t = time.perf_counter()
    engine = get_engine(tuple(langs), target, manual, workers)

with st.sidebar.expander("Models on this device", expanded=False):
    for name in ["silero-vad", "lid-ambernet", "indic-conformer", "whisper-tiny-en",
                 "mt-indic-indic", "mt-indic-en", "mt-en-indic"]:
        mb = dir_mb(MODELS / name)
        st.write(f"{'✅' if mb > 1 else '❌'} `{name}` — {mb:.0f} MB on disk")
    st.write(f"Process RSS: **{rss_mb():.0f} MB**")
    st.write(f"Engine load + warm-up: {engine.load_ms / 1000:.1f} s")

if engine.load_errors.get("indic_model"):
    st.warning("Indic ASR (IndicConformer) is not installed yet — Indic speech will be detected and tagged "
               "but not transcribed. English works end-to-end.", icon="⚠️")
for k, v in engine.load_errors.items():
    if k != "indic_model":
        st.error(f"{k}: {v}")

# ---------------------------------------------------------------- main
st.title("Speech → text in one language")
st.caption("Pipeline: mic → Silero VAD → AmberNet LID ‖ ASR encoder → decode → IndicTrans2 MT → reorder buffer")

tab_mic, tab_browser, tab_file = st.tabs(["🎤 Device mic (live)", "📱 Record in browser", "📁 File / samples"])

with tab_mic:
    st.write("Uses this device's microphone. Speak; segments appear after each pause.")
    c1, c2 = st.columns(2)
    start = c1.button("▶ Start listening", type="primary", width="stretch")
    if c2.button("■ Stop", width="stretch"):
        stop_running_source()
    if start:
        stop_running_source()
        src = MicSource()
        st.session_state["live_source"] = src
        run_stream(engine, src, target)

with tab_browser:
    st.write("Record on any phone/laptop browser connected to this device's network; processing stays on the edge device.")
    rec = st.audio_input("Record")
    if rec is not None and st.button("Transcribe recording", type="primary"):
        audio = load_audio(io.BytesIO(rec.getvalue()))
        run_stream(engine, BufferSource(audio), target, len(audio) / 16000)

with tab_file:
    demos = sorted((ROOT / "data/demo").glob("*.wav"))
    smoke = sorted((ROOT / "data/smoke").glob("*/*.wav"))
    choice = st.selectbox("Sample", ["—", *map(str, demos), *map(str, smoke)],
                          format_func=lambda p: p if p == "—" else str(Path(p).relative_to(ROOT)))
    up = st.file_uploader("…or upload audio", type=["wav", "flac", "mp3", "ogg"])
    path = up if up is not None else (choice if choice != "—" else None)
    if path is not None:
        audio = load_audio(path)
        st.audio(audio, sample_rate=16000)
        ref = Path(str(choice)).with_suffix(".json") if up is None else None
        if ref and ref.exists():
            with st.expander("Reference transcript"):
                st.json(json.loads(ref.read_text()))
        if st.button("Transcribe file", type="primary"):
            run_stream(engine, BufferSource(audio), target, len(audio) / 16000)

# ---------------------------------------------------------------- last run
last = st.session_state.get("last_run")
if last and last["segments"]:
    st.divider()
    segs = last["segments"]
    st.subheader("Transcript")
    for s in segs:
        segment_card(s, target)
    st.subheader("Measured on this device")
    m = st.columns(4)
    m[0].metric("Segments", len(segs))
    if last["audio_s"]:
        m[1].metric("Real-time factor", f"{last['wall_s'] / last['audio_s']:.2f}")
    ttt = [s.timings["time_to_text_ms"] for s in segs if "time_to_text_ms" in s.timings]
    if ttt and not last["audio_s"]:
        m[1].metric("Median time-to-text", f"{np.median(ttt) / 1000:.2f} s")
    m[2].metric("Process RSS", f"{rss_mb():.0f} MB")
    m[3].metric("CPU threads", os.cpu_count())
    st.dataframe(timings_frame(segs), width="stretch")
    st.download_button("Download segments (JSON)",
                       json.dumps([s.to_dict() for s in segs], ensure_ascii=False, indent=1),
                       file_name="segments.json")
