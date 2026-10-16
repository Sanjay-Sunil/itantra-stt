import numpy as np

from indicstt.segment import FLAG_FORCED_CUT
from indicstt.vad import WINDOW, Segmenter, VADConfig


def make_stream(pattern):
    """pattern: list of (is_speech, n_windows) -> audio whose window mean = speech prob."""
    return np.concatenate([np.full(n * WINDOW, 1.0 if s else 0.0, dtype=np.float32) for s, n in pattern])


def run(audio, chunk, cfg=None):
    seg = Segmenter(lambda w: float(w.mean()), cfg)
    out = []
    for i in range(0, len(audio), chunk):
        out += list(seg.push(audio[i : i + chunk]))
    out += list(seg.flush())
    return out


def test_two_segments_in_order():
    audio = make_stream([(0, 20), (1, 40), (0, 30), (1, 25), (0, 30)])
    segs = run(audio, 1600)
    assert [s.seq for s in segs] == [0, 1]
    assert segs[0].start_ms < segs[0].end_ms <= segs[1].start_ms


def test_short_blip_dropped():
    audio = make_stream([(0, 20), (1, 3), (0, 40)])
    assert run(audio, 1600) == []


def test_chunk_size_invariance():
    audio = make_stream([(0, 10), (1, 50), (0, 5), (1, 10), (0, 30), (1, 60), (0, 40)])
    a = run(audio, 160)
    b = run(audio, 16000)
    assert [(s.start_ms, s.end_ms, len(s.audio)) for s in a] == [(s.start_ms, s.end_ms, len(s.audio)) for s in b]


def test_forced_cut():
    cfg = VADConfig(max_segment_s=1.0)
    audio = make_stream([(1, 110), (0, 30)])
    segs = run(audio, 1600, cfg)
    assert len(segs) >= 3
    assert FLAG_FORCED_CUT in segs[0].flags
    assert segs[-1].flags == []


def test_unfinished_speech_flushed():
    audio = make_stream([(0, 5), (1, 30)])
    segs = run(audio, 512)
    assert len(segs) == 1


def test_gain_control_boosts_quiet_and_caps_gain():
    from indicstt.vad import GainControl

    g = GainControl(target=0.3, max_gain=10.0)
    loud = np.full(WINDOW, 0.6, dtype=np.float32)
    assert abs(float(g(loud).max()) - 0.6) < 1e-6  # loud speech passes unchanged (never attenuated)
    quiet = np.full(WINDOW, 0.002, dtype=np.float32)
    for _ in range(300):  # ~10 s: envelope releases toward the quiet level
        out = g(quiet)
    assert float(out.max()) <= 0.002 * 10.0 + 1e-6  # never more than max_gain
    assert float(out.max()) > 0.01  # but clearly boosted
