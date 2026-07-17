"""Hybrid search — dense + BM25, fused with Reciprocal Rank Fusion, then rerank.

The query path, stage by stage::

    question ─▶ embed ─▶ [ dense ‖ BM25 ] ─▶ fuse (RRF) ─▶ rerank? ─▶ top-k

:func:`search_trace` returns a :class:`~contextual_rag.types.RetrievalTrace`
exposing every stage (dense / bm25 / fused / reranked / final) with per-stage
scores and timings — invaluable for debugging *why* a chunk was (or wasn't)
retrieved. :func:`search` is the convenience wrapper returning just the final
list.

The design seam: everything downstream depends only on ``ScoredChunk`` and the
``search()`` contract, so another retriever (a knowledge graph, SQL, an API)
can be added by producing a ranking of chunk ids and joining the RRF fusion —
no rewrite anywhere else.
"""

from __future__ import annotations

import time
from dataclasses import replace

from contextual_rag import index_cache, llm
from contextual_rag import rerank as _rerank
from contextual_rag.bm25 import BM25
from contextual_rag.store import VectorStore
from contextual_rag.types import RetrievalTrace, ScoredChunk

#: The standard RRF constant. 60 keeps head ranks dominant while still
#: rewarding agreement deep in the lists (Cormack et al., 2009).
_RRF_K = 60


def rrf(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: merge rankings without comparing raw scores.

    Each id scores ``sum(1 / (k + rank + 1))`` over the rankings it appears in,
    so an id ranked high by several retrievers rises above one ranked high by
    a single retriever. Rank-based, hence immune to the incomparable score
    scales of BM25 vs cosine similarity.

    Args:
        rankings: One list of ids per retriever, best first.
        k: Dampening constant (see :data:`_RRF_K`).

    Returns:
        id → fused score (higher is better).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search_trace(query: str, *, k: int = 8, domain_id: str = "default",
                 dense_k: int = 50, bm25_k: int = 50, rerank_mode: str = "off",
                 rerank_pool: int = 30, store: VectorStore | None = None,
                 doc_id: str | None = None) -> RetrievalTrace:
    """Run the full hybrid retrieval pipeline, keeping every stage.

    Args:
        query: The question, verbatim.
        k: Number of chunks in ``trace.final``.
        domain_id: Collection to search (ignored when ``store`` is given).
        dense_k: Candidate pool size from the dense retriever.
        bm25_k: Candidate pool size from BM25.
        rerank_mode: ``"off"`` | ``"llm"`` | ``"endpoint"`` — see
            :mod:`contextual_rag.rerank`.
        rerank_pool: How many fused candidates the reranker sees.
        store: Reuse an existing :class:`VectorStore` (recommended — avoids
            re-opening Chroma per query).
        doc_id: Scope the entire search (both retrievers) to one source
            document, so a per-document question can't surface another
            document's passages.

    Returns:
        A :class:`~contextual_rag.types.RetrievalTrace`; ``trace.final`` is
        the answer-ready top-k.
    """
    store = store or VectorStore(domain_id)
    trace = RetrievalTrace(query=query)

    # Records + the corpus-wide BM25 index come from the per-collection cache
    # (rebuilt only when the store changes). Cached records are shared
    # templates — per-query scores are stamped on fresh copies, never on the cache.
    base, corpus_bm = index_cache.get(store)
    if doc_id:
        base = [r for r in base if r.doc_id == doc_id]
    records = [replace(r, score=0.0, scores={}) for r in base]
    by_id = {r.chunk_id: r for r in records}
    if not records:
        return trace

    # Stage 1 — dense: embed the query, cosine-nearest in Chroma.
    t0 = time.perf_counter()
    qvec = llm.embed([query])[0]
    dense_hits = store.query(qvec, dense_k, doc_id=doc_id)
    for d in dense_hits:
        if d.chunk_id in by_id:
            by_id[d.chunk_id].scores["dense"] = d.scores.get("dense", 0.0)
    dense_ranking = [d.chunk_id for d in dense_hits if d.chunk_id in by_id]
    trace.dense = [by_id[c] for c in dense_ranking]
    trace.timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 2 — BM25: the cached corpus index, unless doc-scoped (then a small
    # per-document index, so IDF reflects the searched subset).
    t0 = time.perf_counter()
    bm = BM25().build([(r.chunk_id, r.embed_text) for r in records]) if doc_id else corpus_bm
    bm_hits = bm.search(query, bm25_k)
    for cid, s in bm_hits:
        by_id[cid].scores["bm25"] = s
    bm25_ranking = [cid for cid, _ in bm_hits]
    trace.bm25 = [by_id[c] for c in bm25_ranking]
    trace.timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 3 — fuse with RRF: a chunk found by BOTH retrievers rises.
    fused_scores = rrf([dense_ranking, bm25_ranking])
    for cid, s in fused_scores.items():
        by_id[cid].scores["rrf"] = round(s, 5)
        by_id[cid].score = round(s, 5)
    fused = sorted((by_id[c] for c in fused_scores), key=lambda c: c.score, reverse=True)
    trace.fused = fused

    # Stage 4 — rerank (optional). Skipped when the pool already fits in k:
    # everything would be included regardless, so the call could only reorder,
    # not select — not worth its latency.
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
    """Hybrid search returning just the final top-k (see :func:`search_trace`).

    Args:
        query: The question, verbatim.
        k: Number of chunks to return.
        domain_id: Collection to search (ignored when ``store`` is given).
        rerank_mode: ``"off"`` | ``"llm"`` | ``"endpoint"``.
        store: Reuse an existing :class:`VectorStore`.
        doc_id: Scope the search to one source document.

    Returns:
        The top-k chunks, best first.
    """
    return search_trace(query, k=k, domain_id=domain_id, rerank_mode=rerank_mode,
                        store=store, doc_id=doc_id).final
