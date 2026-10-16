"""Public API: Engine(EngineConfig) wires audio → VAD → scheduler (LID ‖ encoder → decode → MT → reorder)."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioSource, BufferSource, load_audio
from .scheduler import Scheduler, SchedulerConfig
from .segment import FLAG_LOW_CONFIDENCE, FLAG_UNEXPECTED_LANGUAGE, INDIC, LANGUAGES, Segment
from .vad import Segmenter, SileroVAD, VADConfig

FLAG_NO_ASR = "asr_unavailable"
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EngineConfig:
    languages: list[str] = field(default_factory=lambda: ["hi", "en"])  # languages the speaker may use
    target_language: str | None = "en"  # every segment is delivered in this language; None = no MT
    manual_language: str | None = None  # skip LID and force this language
    models_dir: Path = ROOT / "models"
    asr_workers: int = 2
    mt_workers: int = 1
    threads_per_model: int = 4
    low_confidence: float = 0.5
    asr_tier: str = "int8"  # int8 (high-end) | int4mixed (low-end) | fp32 (reference only)
    asr_mode: str = "rnnt"  # rnnt (more accurate) | ctc (faster)
    lid_tier: str = "int8"  # int8 | fp32
    vad: VADConfig = field(default_factory=VADConfig)


class _Stages:
    """Binds the real models to the scheduler's stage interface."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        md = Path(cfg.models_dir)
        self.lid_model = None
        self.indic_model = None
        self.english_model = None
        self.mt = None
        self.load_errors: dict[str, str] = {}
        if not cfg.manual_language and len(cfg.languages) > 1:
            self._try("lid_model", lambda: _load_lid(md, cfg.threads_per_model, cfg.lid_tier))
        if any(lang in INDIC for lang in cfg.languages):
            self._try("indic_model", lambda: _load_indic(md, cfg))
        if "en" in cfg.languages:
            self._try("english_model", lambda: _load_english(md, cfg.threads_per_model))
        if cfg.target_language:
            from .mt import Translator

            self.mt = Translator(md, cfg.threads_per_model)

    def _try(self, name, fn):
        try:
            setattr(self, name, fn())
        except Exception as e:  # noqa: BLE001 - engine keeps running with what is available
            self.load_errors[name] = repr(e)

    # B: LID branch
    def lid(self, seg: Segment) -> Any:
        cfg = self.cfg
        if cfg.manual_language:
            return (cfg.manual_language, 1.0, {}, 0.0)
        if len(cfg.languages) == 1:
            return (cfg.languages[0], 1.0, {}, 0.0)
        if self.lid_model is None:
            return (cfg.languages[0], 0.0, {}, 0.0)
        scores, ms = self.lid_model.scores(seg.audio)
        allowed = {k: v for k, v in scores.items() if k in cfg.languages}
        top_any = max(scores, key=scores.get)
        indic_sel = [x for x in cfg.languages if x in INDIC]
        if any(x not in scores for x in indic_sel):
            # A selected language the LID model has no class for (Odia): only decide English vs Indic here,
            # using the mass on every Indic class; the CTC check in decode() picks the Indic language.
            p_en = scores.get("en", 0.0) if "en" in cfg.languages else 0.0
            p_ind = sum(v for k, v in scores.items() if k in INDIC)
            total = (p_en + p_ind) or 1.0
            if p_en > p_ind:
                return ("en", p_en / total, scores, ms, top_any)
            if len(indic_sel) == 1:
                return (indic_sel[0], p_ind / total, scores, ms, top_any)
            known = {k: v for k, v in allowed.items() if k in INDIC}
            guess = max(known, key=known.get) if known else indic_sel[0]
            return (guess, 0.0, scores, ms, top_any)  # conf 0 → decode() runs the CTC check
        if allowed:
            total = sum(allowed.values())
            lang = max(allowed, key=allowed.get)
            return (lang, allowed[lang] / total, scores, ms, top_any)
        return (cfg.languages[0], 0.0, scores, ms, top_any)

    # B: encoder branch (shared Indic encoder; English has its own model)
    def encode(self, seg: Segment) -> Any:
        if self.indic_model is None:
            return None
        t = time.perf_counter()
        enc = self.indic_model.encode(seg.audio)
        seg.timings["encoder"] = round((time.perf_counter() - t) * 1000, 1)
        return enc

    # C + D: join, choose language, decode
    def decode(self, seg: Segment, lid_result: Any, enc: Any) -> None:
        lang, conf, _scores, lid_ms = lid_result[:4]
        seg.timings["lid"] = round(lid_ms, 1)
        if len(lid_result) > 4 and lid_result[4] not in self.cfg.languages and lid_result[4] in LANGUAGES:
            seg.flags.append(FLAG_UNEXPECTED_LANGUAGE)
        if self.indic_model is not None and enc is not None and hasattr(self.indic_model, "ctc_language_scores"):
            # Languages AmberNet cannot score (Odia) or same-script pairs: let the ASR CTC scores decide
            indic_sel = [x for x in self.cfg.languages if x in INDIC]
            if lang != "en" and len(indic_sel) > 1 and (conf < 0.8 or "or" in indic_sel):
                t = time.perf_counter()
                ctc = self.indic_model.ctc_language_scores(enc, indic_sel)
                seg.timings["lid_ctc"] = round((time.perf_counter() - t) * 1000, 1)
                best = max(ctc, key=ctc.get)
                if best != lang:
                    p = np.exp(np.array(list(ctc.values())) - max(ctc.values()))
                    lang, conf = best, float(p.max() / p.sum())
        seg.language, seg.lid_confidence = lang, round(float(conf), 3)
        if conf < self.cfg.low_confidence and not self.cfg.manual_language and len(self.cfg.languages) > 1:
            seg.flags.append(FLAG_LOW_CONFIDENCE)
        t = time.perf_counter()
        if lang == "en":
            if self.english_model is None:
                seg.flags.append(FLAG_NO_ASR)
                return
            seg.text, tm = self.english_model.transcribe(seg.audio)
            seg.timings["encoder"] = round(tm["encoder"], 1)
            seg.timings["decode"] = round(tm["decode"], 1)
            return
        if self.indic_model is None or enc is None:
            seg.flags.append(FLAG_NO_ASR)
            return
        seg.text = self.indic_model.decode(enc, lang)
        seg.timings["decode"] = round((time.perf_counter() - t) * 1000, 1)

    # F
    def translate(self, seg: Segment, target: str) -> None:
        if self.mt is None or not self.mt.available(seg.language, target):
            return
        seg.translation, ms = self.mt.translate(seg.text, seg.language, target)
        seg.timings["mt"] = round(ms, 1)


def _load_lid(md: Path, threads: int, tier: str):
    from .lid import AmberNetLID

    name = "ambernet.int8.onnx" if tier == "int8" and (md / "lid-ambernet/ambernet.int8.onnx").exists() \
        else "ambernet.onnx"
    return AmberNetLID(md / "lid-ambernet", threads=threads, onnx_name=name)


def _load_english(md: Path, threads: int):
    from .english import WhisperEnglish

    return WhisperEnglish(md / "whisper-tiny-en", threads=threads)


def _load_indic(md: Path, cfg: EngineConfig):
    from .asr import IndicConformerASR

    d = md / ("indic-conformer" if cfg.asr_tier == "fp32" else f"indic-conformer-{cfg.asr_tier}")
    if not d.exists():
        raise FileNotFoundError(f"{d.name} missing: run `python scripts/quantize.py --tier {cfg.asr_tier}`")
    return IndicConformerASR(d, threads=cfg.threads_per_model, mode=cfg.asr_mode,
                             languages=[x for x in cfg.languages if x in INDIC])


class Engine:
    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        bad = [x for x in [*self.cfg.languages, self.cfg.target_language] if x and x not in LANGUAGES]
        if bad:
            raise ValueError(f"unsupported language(s): {bad}")
        t = time.perf_counter()
        self.stages = _Stages(self.cfg)
        self.vad_model = SileroVAD(str(Path(self.cfg.models_dir) / "silero-vad/onnx/model.onnx"))
        self.warmup()
        self.load_ms = (time.perf_counter() - t) * 1000
        self.events: queue.Queue = queue.Queue()
        self._source: AudioSource | None = None

    def warmup(self) -> None:
        """Load lazily-loaded models and run each once so the first real segment is not slow."""
        st, tgt = self.stages, self.cfg.target_language
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        if st.lid_model:
            st.lid_model.scores(noise)
        if st.english_model:
            st.english_model.transcribe(noise, max_tokens=2)
        if st.mt and tgt:
            for src in self.cfg.languages:
                if src != tgt and st.mt.available(src, tgt):
                    st.mt.translate("नमस्ते" if src != "en" else "hello", src, tgt)

    @property
    def load_errors(self) -> dict[str, str]:
        return self.stages.load_errors

    def stream(self, source: AudioSource) -> Iterator[Segment]:
        """Run the pipeline on a live or file source; yields segments in speaking order."""
        self._source = source
        self.vad_model.reset()
        segmenter = Segmenter(self._timed_vad, self.cfg.vad)
        sched = Scheduler(self.stages, SchedulerConfig(
            asr_workers=self.cfg.asr_workers, mt_workers=self.cfg.mt_workers,
            target_language=self.cfg.target_language,
        ))
        sched.events = self.events
        sched.start()
        self._vad_ms = 0.0

        def vad_thread():  # A: never waits on downstream stages
            try:
                for chunk in source.chunks():
                    for seg in segmenter.push(chunk):
                        self._submit(sched, seg)
                for seg in segmenter.flush():
                    self._submit(sched, seg)
            finally:
                sched.close(wait=True)

        threading.Thread(target=vad_thread, name="vad", daemon=True).start()
        yield from sched.results()

    def _timed_vad(self, window: np.ndarray) -> float:
        t = time.perf_counter()
        p = self.vad_model(window)
        self._vad_ms += (time.perf_counter() - t) * 1000
        return p

    def _submit(self, sched: Scheduler, seg: Segment) -> None:
        seg.timings["vad"] = round(self._vad_ms, 1)  # VAD compute spent on this segment's span
        self._vad_ms = 0.0
        seg.timings["finalized_at"] = time.time()
        sched.submit(seg)

    def stop(self) -> None:
        if self._source:
            self._source.stop()

    def transcribe_file(self, path) -> list[Segment]:
        return list(self.stream(BufferSource(load_audio(path))))

    def transcribe_audio(self, audio: np.ndarray) -> list[Segment]:
        return list(self.stream(BufferSource(audio)))
