"""Per-collection corpus cache — records + BM25 index reused across queries.

Without it, every query would pull the whole Chroma collection and rebuild the BM25
index from scratch: seconds of pure overhead per question once a corpus grows. The
corpus only changes on ingestion, so both are cached per (store path, collection) and
dropped when a write invalidates them. The collection count is re-checked on every hit,
catching writes this process didn't see (another process on the same store).

Cached records are templates — search must not mutate them (it stamps per-query scores
on fresh copies, see search.py).
"""

from __future__ import annotations

from threading import Lock

from contextual_rag.bm25 import BM25
from contextual_rag.types import ScoredChunk

_lock = Lock()
_cache: dict[tuple, tuple[int, list[ScoredChunk], BM25]] = {}


def get(store) -> tuple[list[ScoredChunk], BM25]:
    """The store's full record list and corpus-wide BM25 index, cached until stale."""
    count = store.count()
    with _lock:
        hit = _cache.get(store.cache_key)
    if hit and hit[0] == count:
        return hit[1], hit[2]
    records = store.all_records()
    index = BM25().build([(r.chunk_id, r.embed_text) for r in records])
    with _lock:
        _cache[store.cache_key] = (count, records, index)
    return records, index


def invalidate(cache_key: tuple) -> None:
    with _lock:
        _cache.pop(cache_key, None)
