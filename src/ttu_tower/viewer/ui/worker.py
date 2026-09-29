"""Background jobs: run a function on a worker thread and deliver its result
(or error) back on the GUI thread. A newer job under the same key supersedes
an older one: one that hasn't started yet skips its work when its turn comes,
and one already running has its result dropped - that's how navigating away
"cancels" work. Jobs someone is waiting on (an inspector opening) can jump the
queue with a priority.

Nothing Qt may be created or destroyed off the GUI thread. The workers are
plain Python threads (a QThreadPool deletes its QRunnables on its own threads),
results come back through a queued signal, and Python's cycle collector runs
only on the GUI thread: left automatic, it runs on whichever thread happens to
allocate, and a worker freeing a closed window's or a cleared plot's Qt objects
deadlocked PySide now and then.
"""
import itertools
import logging
import queue
import threading
import traceback

from pyqtgraph.util.garbage_collector import GarbageCollector
from PySide6.QtCore import QObject, Qt, Signal, Slot

_SKIPPED = object()  # a superseded job's result: nothing was computed
_collector = None


def collect_garbage_on_gui_thread() -> None:
    """Turn off automatic cycle collection and collect on a GUI-thread timer instead."""
    global _collector
    if _collector is None:
        _collector = GarbageCollector(interval=0.5)


class _Signals(QObject):
    finished = Signal(int, bool, object)


class _Pool:
    """Worker threads taking jobs from a priority queue (highest first, then in order)."""

    def __init__(self, threads: int):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._order = itertools.count()
        self._cond = threading.Condition()
        self._pending = self._active = 0
        self._threads = [threading.Thread(target=self._work, name=f"ttu-view-worker-{i}", daemon=True)
                         for i in range(threads)]
        for t in self._threads:
            t.start()

    def start(self, job, priority: int = 0) -> None:
        with self._cond:
            self._pending += 1
        self._queue.put((-priority, next(self._order), job))

    def _work(self) -> None:
        while True:
            _, _, job = self._queue.get()
            if job is None:
                return
            with self._cond:
                self._pending -= 1
                self._active += 1
            try:
                job()
            finally:
                with self._cond:
                    self._active -= 1
                    self._cond.notify_all()

    def activeThreadCount(self) -> int:
        return self._active

    def waitForDone(self, msecs: int = 30_000) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0 and self._active == 0, timeout=msecs / 1000)

    def shutdown(self) -> None:
        for _ in self._threads:
            self._queue.put((float("inf"), next(self._order), None))


class JobRunner(QObject):
    """Owned by the GUI thread; `busyChanged` reports how many jobs are live."""

    busyChanged = Signal(int, str)

    def __init__(self, max_threads: int = 2, parent=None):
        super().__init__(parent)
        collect_garbage_on_gui_thread()
        self.pool = _Pool(max_threads)
        self._signals = _Signals()
        self._signals.finished.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        self._ids = itertools.count(1)
        self._callbacks: dict[int, tuple] = {}
        self._latest: dict[str, int] = {}
        self._superseded: set[int] = set()  # read by the worker threads
        self._lock = threading.Lock()

    def submit(self, fn, *args, on_done=None, on_error=None, key: str | None = None, label: str = "",
               priority: int = 0, **kwargs) -> int:
        job_id = next(self._ids)
        self._callbacks[job_id] = (on_done, on_error, key, label)
        if key is not None:
            older = self._latest.get(key)
            if older is not None:
                with self._lock:
                    self._superseded.add(older)
            self._latest[key] = job_id
        self.pool.start(lambda: self._run(job_id, fn, args, kwargs), priority)
        self._report()
        return job_id

    def _run(self, job_id: int, fn, args, kwargs) -> None:
        """On a worker thread."""
        with self._lock:
            skip = job_id in self._superseded  # superseded while it waited in the queue
        if skip:
            self._signals.finished.emit(job_id, True, _SKIPPED)
            return
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # delivered to on_error on the GUI thread
            self._signals.finished.emit(job_id, False, (exc, traceback.format_exc()))
            return
        self._signals.finished.emit(job_id, True, result)

    def wait(self, msecs: int = 30_000) -> bool:
        return self.pool.waitForDone(msecs)

    def shutdown(self) -> None:
        """Let the worker threads end once the queued jobs are done."""
        self.pool.shutdown()

    @Slot(int, bool, object)
    def _deliver(self, job_id: int, ok: bool, payload):
        with self._lock:
            self._superseded.discard(job_id)
        on_done, on_error, key, _ = self._callbacks.pop(job_id, (None, None, None, ""))
        superseded = key is not None and self._latest.get(key) != job_id
        if key is not None and self._latest.get(key) == job_id:
            del self._latest[key]
        self._report()
        if superseded or payload is _SKIPPED:
            return
        if ok and on_done is not None:
            on_done(payload)
        elif not ok:
            exc, tb = payload
            if on_error is not None:
                on_error(exc, tb)
            else:
                logging.getLogger("ttu_view").error(tb)

    def _report(self):
        labels = [label for (_, _, _, label) in self._callbacks.values() if label]
        self.busyChanged.emit(len(self._callbacks), labels[-1] if labels else "")
