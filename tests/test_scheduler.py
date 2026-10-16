import random
import time

from indicstt.scheduler import Scheduler, SchedulerConfig
from indicstt.segment import FLAG_ERROR, FLAG_TIMEOUT, Segment


class FakeStages:
    def __init__(self, rng, max_delay=0.0005, hang_seq=None, fail_seq=None):
        self.rng, self.max_delay, self.hang_seq, self.fail_seq = rng, max_delay, hang_seq, fail_seq
        self.translated = []

    def _sleep(self):
        time.sleep(self.rng.random() * self.max_delay)

    def lid(self, seg):
        self._sleep()
        return ["hi", "ta", "en"][seg.seq % 3]

    def encode(self, seg):
        if seg.seq == self.hang_seq:
            time.sleep(1.0)
        if seg.seq == self.fail_seq:
            raise RuntimeError("boom")
        self._sleep()
        return seg.seq

    def decode(self, seg, lid, enc):
        self._sleep()
        seg.language, seg.text = lid, f"text-{enc}"

    def translate(self, seg, target):
        self._sleep()
        self.translated.append(seg.seq)
        seg.translation = f"{seg.text}->{target}"


def run(stages, n, cfg):
    s = Scheduler(stages, cfg)
    s.start()
    for i in range(n):
        s.submit(Segment(seq=i, start_ms=i * 1000, end_ms=i * 1000 + 500))
    s.close(wait=True)
    return list(s.results())


def test_in_order_and_skip_logic():
    st = FakeStages(random.Random(0))
    out = run(st, 30, SchedulerConfig(asr_workers=3, mt_workers=2, target_language="hi"))
    assert [s.seq for s in out] == list(range(30))
    for s in out:
        if s.language == "hi":
            assert s.translation is None and s.seq not in st.translated
        else:
            assert s.translation == f"text-{s.seq}->hi"


def test_randomized_no_deadlock_always_ordered():
    rng = random.Random(42)
    for _ in range(1000):
        n = rng.randint(1, 12)
        cfg = SchedulerConfig(asr_workers=rng.randint(1, 3), mt_workers=rng.randint(1, 2),
                              queue_size=rng.randint(1, 4), target_language=rng.choice([None, "hi", "en"]))
        out = run(FakeStages(rng, max_delay=0.0002), n, cfg)
        assert [s.seq for s in out] == list(range(n))


def test_timeout_releases_stuck_segment():
    st = FakeStages(random.Random(1), hang_seq=1)
    s = Scheduler(st, SchedulerConfig(asr_workers=2, reorder_timeout_s=0.2))
    s.start()
    for i in range(4):
        s.submit(Segment(seq=i, start_ms=0, end_ms=1))
    s.close(wait=True)
    out = list(s.results())
    assert [x.seq for x in out] == [0, 1, 2, 3]
    assert FLAG_TIMEOUT in out[1].flags and out[1].text == ""


def test_stage_error_does_not_stop_stream():
    st = FakeStages(random.Random(2), fail_seq=2)
    out = run(st, 5, SchedulerConfig())
    assert [x.seq for x in out] == list(range(5))
    assert FLAG_ERROR in out[2].flags
