import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.models,
              pytest.mark.skipif(not os.environ.get("INDICSTT_MODELS"), reason="set INDICSTT_MODELS=1")]


def test_english_file_end_to_end():
    from indicstt.engine import Engine, EngineConfig

    clip = next((ROOT / "data/smoke/en").glob("*.wav"))
    segs = Engine(EngineConfig(languages=["en"], target_language="hi")).transcribe_file(clip)
    assert segs and [s.seq for s in segs] == list(range(len(segs)))
    assert all(s.language == "en" and s.text and s.translation for s in segs)
