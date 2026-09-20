"""Queue-based multiprocess logging: a JSON-lines (Cutelog-compatible)
formatter, a listener for the main process, worker configuration, the
warnings bridge, and per-record pipeline context (stage, unit, batch, boom,
half_hour).
"""
import json
import logging
import logging.handlers
import sys
import time
import traceback
import warnings
from multiprocessing import Queue

CONTEXT_FIELDS = ("stage", "unit", "batch", "boom", "half_hour")


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line; Cutelog-compatible."""

    _BASE_KEYS = {
        "created", "asctime", "name", "levelname", "levelno", "pathname",
        "lineno", "msg", "process", "thread", "exc_info", "exc_text",
        "stack_info", "msecs", "relativeCreated", "funcName", "module",
        "filename", "processName", "threadName", "args", "taskName", "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "created": getattr(record, "created", time.time()),
            "asctime": self.formatTime(record, self.datefmt),
            "name": record.name,
            "levelname": record.levelname,
            "levelno": record.levelno,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "msg": record.getMessage(),
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            log_obj["exc_text"] = "".join(traceback.format_exception(*record.exc_info))
        for k, v in record.__dict__.items():
            if k not in self._BASE_KEYS and not k.startswith("_"):
                log_obj[k] = v
        return json.dumps(log_obj, ensure_ascii=False)


class ContextAdapter(logging.LoggerAdapter):
    """Injects the fixed context fields into every record."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        for field in CONTEXT_FIELDS:
            extra.setdefault(field, self.extra.get(field))
        return msg, kwargs


def get_logger(name: str, **context) -> ContextAdapter:
    """A logger that stamps every record with the given context fields
    (stage, unit, batch, boom, half_hour), null where not given.
    """
    return ContextAdapter(logging.getLogger(name), context)


def listener_configurer(logfile: str):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setFormatter(JsonLinesFormatter())
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)


def log_listener(queue: Queue, logfile: str):
    """Runs in the main process: drains the queue and writes to logfile until
    a None sentinel is received.
    """
    listener_configurer(logfile)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logging.getLogger(record.name).handle(record)
        except Exception:
            print("Logging listener error:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def configure_worker(queue: Queue):
    """Runs once in each worker process: routes all logging through the queue."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.DEBUG)
    enable_warning_bridge()


def enable_warning_bridge():
    """Routes warnings.warn(...) into the "MAIN" logger as WARNING records."""
    log = logging.getLogger("MAIN")

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        log.warning(
            "python_warning",
            extra={
                "warning_message": str(message),
                "warning_category": getattr(category, "__name__", str(category)),
                "warning_filename": filename,
                "warning_lineno": lineno,
                "warning_line": line,
            },
        )

    warnings.showwarning = _showwarning
