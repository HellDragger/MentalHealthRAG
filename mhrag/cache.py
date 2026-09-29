"""Small thread-safe LRU caches for query embeddings, retrieval results and (optionally) responses."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


class LRUCache:
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        return {"size": len(self._d), "hits": self.hits, "misses": self.misses}
