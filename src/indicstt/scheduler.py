"""Concurrency layer: queue/parallel points A–G from agent-kit/docs/architecture.md §3.

A  VAD thread → bounded segment queue
B  per segment, LID and the ASR encoder start together
C  join: decode starts when both are done
D  ASR worker pool (2–3 workers)
E  only segments whose language ≠ target enter the MT queue
F  MT worker pool
G  reorder buffer: release N only after N−1; a stuck segment is released with flag `timeout`

Stages are injected, so this module is testable with fakes and knows nothing about models.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .segment import FLAG_ERROR, FLAG_TIMEOUT, Segment

_STOP = object()


class Stages(Protocol):
    def lid(self, seg: Segment) -> Any: ...
    def encode(self, seg: Segment) -> Any: ...
    def decode(self, seg: Segment, lid_result: Any, enc_result: Any) -> None:
        """Fill seg.language, seg.lid_confidence, seg.text (and timings/flags)."""
    def translate(self, seg: Segment, target: str) -> None:
        """Fill seg.translation."""


@dataclass
class SchedulerConfig:
    asr_workers: int = 2
    mt_workers: int = 1
    queue_size: int = 32
    reorder_timeout_s: float = 20.0
    target_language: str | None = None


class ReorderBuffer:
    """Releases segments strictly in seq order (point G)."""

    def __init__(self, timeout_s: float, emit: Callable[[Segment], None]):
        self.timeout_s = timeout_s
        self.emit = emit
        self.next_seq = 0
        self.pending: dict[int, Segment] = {}
        self.started: dict[int, tuple[float, Segment]] = {}  # seq -> (t_enqueued, seg)
        self.lock = threading.Lock()

    def register(self, seg: Segment) -> None:
        with self.lock:
            self.started[seg.seq] = (time.monotonic(), seg)

    def put(self, seg: Segment) -> None:
        with self.lock:
            self.started.pop(seg.seq, None)
            if seg.seq < self.next_seq:
                return  # already released by timeout; drop the late result
            seg.timings["wait_reorder"] = time.monotonic()  # converted to ms on release
            self.pending[seg.seq] = seg
            self._drain()

    def check_timeouts(self) -> None:
        with self.lock:
            entry = self.started.get(self.next_seq)
            if entry and self.pending and time.monotonic() - entry[0] > self.timeout_s:
                _, stuck = entry
                del self.started[self.next_seq]
                placeholder = Segment(seq=stuck.seq, start_ms=stuck.start_ms, end_ms=stuck.end_ms,
                                      flags=[*stuck.flags, FLAG_TIMEOUT])
                self.pending[stuck.seq] = placeholder
                self._drain()

    def _drain(self) -> None:
        now = time.monotonic()
        while self.next_seq in self.pending:
            seg = self.pending.pop(self.next_seq)
            t = seg.timings.get("wait_reorder")
            seg.timings["wait_reorder"] = round((now - t) * 1000, 1) if t else 0.0
            self.next_seq += 1
            self.emit(seg)


class Scheduler:
    def __init__(self, stages: Stages, config: SchedulerConfig | None = None):
        self.stages = stages
        self.cfg = config or SchedulerConfig()
        self.seg_q: queue.Queue = queue.Queue(self.cfg.queue_size)  # A
        self.mt_q: queue.Queue = queue.Queue(self.cfg.queue_size)  # E
        self.out_q: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self.reorder = ReorderBuffer(self.cfg.reorder_timeout_s, self.out_q.put)
        # B: LID runs beside the encoder; one LID thread per ASR worker keeps them truly parallel
        self._lid_pool = ThreadPoolExecutor(max(1, self.cfg.asr_workers), thread_name_prefix="lid")
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._started = False

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(self.cfg.asr_workers):
            self._spawn(self._asr_worker, f"asr-{i}")
        for i in range(self.cfg.mt_workers):
            self._spawn(self._mt_worker, f"mt-{i}")
        self._spawn(self._watchdog, "reorder-watchdog")

    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def submit(self, seg: Segment) -> None:
        """Called from the VAD thread (point A). Never drops audio: blocks, but reports it."""
        self.reorder.register(seg)
        try:
            self.seg_q.put_nowait(seg)
        except queue.Full:
            self.events.put({"event": "falling_behind", "seq": seg.seq, "queue": "segment"})
            self.seg_q.put(seg)

    def close(self, wait: bool = True) -> None:
        """No more input. With wait=True, blocks until every submitted segment is released."""
        for _ in range(self.cfg.asr_workers):
            self.seg_q.put(_STOP)
        if wait:
            for t in self._threads:
                if t.name.startswith("asr"):
                    t.join()
            for _ in range(self.cfg.mt_workers):
                self.mt_q.put(_STOP)
            for t in self._threads:
                if t.name.startswith("mt"):
                    t.join()
        self._stopping.set()
        self.out_q.put(_STOP)
        self._lid_pool.shutdown(wait=False)

    def results(self) -> Iterator[Segment]:
        while True:
            item = self.out_q.get()
            if item is _STOP:
                return
            yield item

    # ---- workers ---------------------------------------------------------
    def _asr_worker(self) -> None:
        while True:
            seg = self.seg_q.get()
            if seg is _STOP:
                return
            try:
                lid_future = self._lid_pool.submit(self.stages.lid, seg)  # B
                enc = self.stages.encode(seg)
                lid = lid_future.result()  # C (join)
                self.stages.decode(seg, lid, enc)  # D
            except Exception as e:  # noqa: BLE001 - one bad segment must not stop the stream
                seg.flags.append(FLAG_ERROR)
                self.events.put({"event": "error", "seq": seg.seq, "stage": "asr", "error": repr(e)})
                self.reorder.put(seg)
                continue
            target = self.cfg.target_language
            if target and seg.language and seg.language != target and seg.text.strip():
                seg.target_language = target
                try:
                    self.mt_q.put_nowait(seg)  # E
                except queue.Full:
                    self.events.put({"event": "falling_behind", "seq": seg.seq, "queue": "mt"})
                    self.mt_q.put(seg)
            else:
                seg.target_language = target
                self.reorder.put(seg)  # G (MT skipped)

    def _mt_worker(self) -> None:
        while True:
            seg = self.mt_q.get()
            if seg is _STOP:
                return
            try:
                self.stages.translate(seg, seg.target_language)  # F
            except Exception as e:  # noqa: BLE001
                seg.flags.append(FLAG_ERROR)
                self.events.put({"event": "error", "seq": seg.seq, "stage": "mt", "error": repr(e)})
            self.reorder.put(seg)

    def _watchdog(self) -> None:
        while not self._stopping.wait(0.25):
            self.reorder.check_timeouts()
