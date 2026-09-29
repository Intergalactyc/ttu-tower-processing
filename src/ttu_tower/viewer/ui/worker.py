"""Background jobs: run a function on a thread pool and deliver its result
(or error) back on the GUI thread. A newer job under the same key supersedes
an older one: still queued, it's dropped; already running, its result is -
that's how navigating away "cancels" work. Jobs someone is waiting on (an
inspector opening) can jump the queue with a priority.
"""
import itertools
import logging
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot


class _Signals(QObject):
    finished = Signal(int, bool, object)


class _Job(QRunnable):
    def __init__(self, job_id: int, fn, args, kwargs, signals: _Signals):
        super().__init__()
        self.job_id, self.fn, self.args, self.kwargs, self.signals = job_id, fn, args, kwargs, signals

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # delivered to on_error on the GUI thread
            self.signals.finished.emit(self.job_id, False, (exc, traceback.format_exc()))
            return
        self.signals.finished.emit(self.job_id, True, result)


class JobRunner(QObject):
    """Owned by the GUI thread; `busyChanged` reports how many jobs are live."""

    busyChanged = Signal(int, str)

    def __init__(self, max_threads: int = 2, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max_threads)
        self._signals = _Signals()
        self._signals.finished.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        self._ids = itertools.count(1)
        self._callbacks: dict[int, tuple] = {}
        self._latest: dict[str, int] = {}
        self._jobs: dict[int, _Job] = {}  # kept alive until delivered, so a queued one can be taken back

    def submit(self, fn, *args, on_done=None, on_error=None, key: str | None = None, label: str = "",
               priority: int = 0, **kwargs) -> int:
        job_id = next(self._ids)
        self._callbacks[job_id] = (on_done, on_error, key, label)
        if key is not None:
            self._withdraw(self._latest.get(key))
            self._latest[key] = job_id
        job = _Job(job_id, fn, args, kwargs, self._signals)
        job.setAutoDelete(False)
        self._jobs[job_id] = job
        self.pool.start(job, priority)
        self._report()
        return job_id

    def _withdraw(self, job_id: int | None) -> None:
        """Drop a superseded job that hasn't started yet."""
        job = self._jobs.get(job_id)
        if job is not None and self.pool.tryTake(job):
            self._jobs.pop(job_id, None)
            self._callbacks.pop(job_id, None)

    def wait(self, msecs: int = 30_000) -> bool:
        return self.pool.waitForDone(msecs)

    @Slot(int, bool, object)
    def _deliver(self, job_id: int, ok: bool, payload):
        self._jobs.pop(job_id, None)
        on_done, on_error, key, _ = self._callbacks.pop(job_id, (None, None, None, ""))
        superseded = key is not None and self._latest.get(key) != job_id
        if key is not None and self._latest.get(key) == job_id:
            del self._latest[key]
        self._report()
        if superseded:
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
