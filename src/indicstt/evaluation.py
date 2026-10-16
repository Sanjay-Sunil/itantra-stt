"""WER / CER with Indic-aware normalisation, and the regression gate (agent-kit D12)."""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("।", " ").replace("॥", " ")
    text = "".join(ch if unicodedata.category(ch)[0] in "LMN" or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def edit_distance(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[-1]


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return edit_distance(r, h), len(r)


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(normalize(ref).replace(" ", "")), list(normalize(hyp).replace(" ", ""))
    return edit_distance(r, h), len(r)


def gate(result: dict[str, float], baseline: dict[str, float], max_loss: float = 2.0) -> list[str]:
    """Languages whose WER rose by more than max_loss absolute points. Empty list = pass."""
    return [k for k, v in result.items() if k in baseline and v - baseline[k] > max_loss]
