"""Segment record: the engine's output contract (agent-kit/docs/architecture.md §4.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

LANGUAGES = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "or": "Odia",
    "bn": "Bengali",
    "en": "English",
}
INDIC = [c for c in LANGUAGES if c != "en"]

FLAG_UNEXPECTED_LANGUAGE = "unexpected_language"
FLAG_FORCED_CUT = "forced_cut"
FLAG_TIMEOUT = "timeout"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_ERROR = "error"


@dataclass
class Segment:
    seq: int
    start_ms: int
    end_ms: int
    language: str = ""
    lid_confidence: float = 0.0
    text: str = ""
    translation: str | None = None
    target_language: str | None = None
    tokens: list | None = None
    timings: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    # internal: raw audio, dropped from to_dict()
    audio: np.ndarray | None = field(default=None, repr=False)

    @property
    def output_text(self) -> str:
        """Text in the target language: the translation if MT ran, otherwise the transcript."""
        return self.translation if self.translation is not None else self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        return d
