"""Per-collection corpus cache — records + BM25 index reused across queries.

Without it, every query would pull the whole Chroma collection and rebuild the
BM25 index from scratch: seconds of pure overhead per question once a corpus
grows. The corpus only changes on ingestion, so both artifacts are cached per
``(store path, collection)`` and dropped when a write invalidates them.

Freshness model, stated honestly:

- **In-process writes** always invalidate (every ``VectorStore`` write calls
  :func:`invalidate`), and a generation counter prevents a concurrent reader
  from re-caching a snapshot taken just before the write.
- **Other processes** on the same on-disk store are detected via a collection
  count re-check on every hit — which catches adds and deletes, but NOT a
  same-count upsert (re-ingesting an edited document rewrites the same chunk
  ids). If multiple writer processes share a store, restart readers — or add
  your own signal — after cross-process re-ingestion.

Cached records are **templates** — search must never mutate them; it stamps
per-query scores on fresh copies (see :mod:`contextual_rag.search`).
"""

from __future__ import annotations

from threading import Lock

from contextual_rag.bm25 import BM25
from contextual_rag.types import ScoredChunk

_lock = Lock()
_cache: dict[tuple, tuple[int, list[ScoredChunk], BM25]] = {}
#: Bumped by invalidate(); lets get() detect a write that raced its rebuild.
_generation: dict[tuple, int] = {}


def get(store) -> tuple[list[ScoredChunk], BM25]:
    """Return the store's full record list and corpus-wide BM25 index.

    Served from cache while fresh; rebuilt when the collection count changed
    or a write invalidated the entry. A rebuild that lost a race with an
    invalidation returns its (still correct) fresh data but does not populate
    the cache — the next call rebuilds again rather than serving a snapshot
    that predates the write.

    Args:
        store: A :class:`~contextual_rag.store.VectorStore` (anything with
            ``cache_key``, ``count()`` and ``all_records()``).

    Returns:
        ``(records, bm25_index)`` — treat both as read-only.
    """
    count = store.count()
    with _lock:
        hit = _cache.get(store.cache_key)
        gen = _generation.get(store.cache_key, 0)
    if hit and hit[0] == count:
        return hit[1], hit[2]
    records = store.all_records()
    index = BM25().build([(r.chunk_id, r.embed_text) for r in records])
    with _lock:
        if _generation.get(store.cache_key, 0) == gen:  # no write raced us
            _cache[store.cache_key] = (count, records, index)
    return records, index


def invalidate(cache_key: tuple) -> None:
    """Drop the cache entry for one collection (called on every store write)."""
    with _lock:
        _cache.pop(cache_key, None)
        _generation[cache_key] = _generation.get(cache_key, 0) + 1
