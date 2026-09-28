"""A small thread-safe LRU cache whose concurrent requests for one key share
a single computation.
"""
import threading
from collections import OrderedDict
from concurrent.futures import Future


class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._inflight: dict = {}
        self._lock = threading.Lock()

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key, compute):
        """The cached value for `key`, computing it with `compute()` on a miss.
        A caller asking for a key another thread is already computing waits
        for that result instead of computing it again.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[key] = future
        if not owner:
            return future.result()
        try:
            value = compute()
        except BaseException as exc:
            with self._lock:
                del self._inflight[key]
            future.set_exception(exc)
            raise
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            del self._inflight[key]
        future.set_result(value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
