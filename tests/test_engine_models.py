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


def test_shared_engine_concurrent_requests_match_sequential():
    """Public UI: one loaded engine, several visitors at once with different options."""
    import threading

    from indicstt.engine import Engine, EngineConfig

    base = Engine(EngineConfig(languages=["hi", "ta", "en"], target_language=None))
    hi = next((ROOT / "data/smoke/hi").glob("*.wav"))
    ta = next((ROOT / "data/smoke/ta").glob("*.wav"))
    jobs = [(base.derive(languages=["hi", "en"]), hi), (base.derive(languages=["ta", "en"]), ta)]
    alone = [[s.text for s in e.transcribe_file(f)] for e, f in jobs]
    together = [None, None]

    def run(i):
        e, f = jobs[i]
        together[i] = [s.text for s in e.derive().transcribe_file(f)]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert together == alone
    assert all(alone)
