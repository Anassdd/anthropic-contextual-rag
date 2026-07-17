"""Hybrid search — dense + BM25, fused with Reciprocal Rank Fusion, then optional rerank.

Returns a `RetrievalTrace` exposing every stage (dense / bm25 / fused / reranked / final),
for debugging and step-by-step UIs. `search()` is the convenience that returns just the
final list. Another retriever (e.g. a knowledge graph) can be added behind this seam by
returning `ScoredChunk`s and joining the fusion.
"""

from __future__ import annotations

import time
from dataclasses import replace

from contextual_rag import index_cache, llm
from contextual_rag import rerank as _rerank
from contextual_rag.bm25 import BM25
from contextual_rag.store import VectorStore
from contextual_rag.types import RetrievalTrace, ScoredChunk

_RRF_K = 60


def rrf(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: an id ranked high by several retrievers rises."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search_trace(query: str, *, k: int = 8, domain_id: str = "default",
                 dense_k: int = 50, bm25_k: int = 50, rerank_mode: str = "off",
                 rerank_pool: int = 30, store: VectorStore | None = None,
                 doc_id: str | None = None) -> RetrievalTrace:
    store = store or VectorStore(domain_id)
    trace = RetrievalTrace(query=query)

    # Records + the corpus-wide BM25 index come from the per-collection cache (rebuilt
    # only when the store changes). The cached records are shared templates — per-query
    # scores are stamped on fresh copies, never on the cache.
    base, corpus_bm = index_cache.get(store)
    # doc_id scopes the whole search to one source document (both dense and BM25
    # candidate pools), so a per-document question can't retrieve another document's
    # passages.
    if doc_id:
        base = [r for r in base if r.doc_id == doc_id]
    records = [replace(r, score=0.0, scores={}) for r in base]
    by_id = {r.chunk_id: r for r in records}
    if not records:
        return trace

    # dense
    t0 = time.perf_counter()
    qvec = llm.embed([query])[0]
    dense_hits = store.query(qvec, dense_k, doc_id=doc_id)
    for d in dense_hits:
        if d.chunk_id in by_id:
            by_id[d.chunk_id].scores["dense"] = d.scores.get("dense", 0.0)
    dense_ranking = [d.chunk_id for d in dense_hits if d.chunk_id in by_id]
    trace.dense = [by_id[c] for c in dense_ranking]
    trace.timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # bm25 — the cached corpus index, unless doc-scoped (then a small per-doc index, so
    # IDF reflects the searched subset)
    t0 = time.perf_counter()
    bm = BM25().build([(r.chunk_id, r.embed_text) for r in records]) if doc_id else corpus_bm
    bm_hits = bm.search(query, bm25_k)
    for cid, s in bm_hits:
        by_id[cid].scores["bm25"] = s
    bm25_ranking = [cid for cid, _ in bm_hits]
    trace.bm25 = [by_id[c] for c in bm25_ranking]
    trace.timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # fuse (RRF)
    fused_scores = rrf([dense_ranking, bm25_ranking])
    for cid, s in fused_scores.items():
        by_id[cid].scores["rrf"] = round(s, 5)
        by_id[cid].score = round(s, 5)
    fused = sorted((by_id[c] for c in fused_scores), key=lambda c: c.score, reverse=True)
    trace.fused = fused

    # rerank (optional) — skipped when the pool fits in k anyway: everything would be
    # included regardless, so the call would only reorder, not select.
    if rerank_mode != "off" and len(fused) > k:
        t0 = time.perf_counter()
        trace.reranked = _rerank.rerank(query, fused[:rerank_pool], mode=rerank_mode)
        trace.reranked_applied = True
        trace.timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        trace.final = trace.reranked[:k]
    else:
        trace.reranked = fused
        trace.final = fused[:k]

    return trace


def search(query: str, *, k: int = 8, domain_id: str = "default",
           rerank_mode: str = "off", store: VectorStore | None = None,
           doc_id: str | None = None) -> list[ScoredChunk]:
    return search_trace(query, k=k, domain_id=domain_id, rerank_mode=rerank_mode,
                        store=store, doc_id=doc_id).final
