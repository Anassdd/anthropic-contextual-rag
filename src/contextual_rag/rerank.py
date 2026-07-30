"""Reranker seam — reorder fused candidates by true relevance to the query.

Retrieval scores (cosine, BM25, RRF) measure *similarity*; a reranker reads
query and passage together and judges *relevance* — which is why it is the
single biggest add-on win in Anthropic's published results (retrieval failures
−49% → −67% when a reranker is added).

Three modes, all GPU-free:

- ``"off"`` — pass-through (the "optional, gracefully skipped" default).
- ``"llm"`` — RankGPT-style listwise ranking: ONE chat call ranks all
  candidates. Works through the existing endpoint, no extra service.
- ``"endpoint"`` — a dedicated hosted cross-encoder (Cohere/Jina request
  shape), used when ``RERANK_MODEL`` + ``RERANK_BASE_URL`` are configured.

Both real modes degrade gracefully: candidates the reranker forgot to mention
are appended in their original order, so a flaky model can reorder results but
never lose them.
"""

from __future__ import annotations

import json
import re
import urllib.request

from contextual_rag import llm
from contextual_rag.config import get_settings
from contextual_rag.types import ScoredChunk


def endpoint_configured() -> bool:
    """True if a dedicated reranker endpoint is configured."""
    s = get_settings()
    return bool(s.rerank_model and s.rerank_base_url)


def rerank(query: str, chunks: list[ScoredChunk], *, mode: str = "llm",
           top_k: int | None = None) -> list[ScoredChunk]:
    """Reorder ``chunks`` by relevance to ``query``.

    Args:
        query: The question, verbatim.
        chunks: Fused candidates, current-best first.
        mode: ``"off"`` | ``"llm"`` | ``"endpoint"``. ``"endpoint"`` falls back
            to ``"llm"`` when no endpoint is configured.
        top_k: Truncate the result (None = keep all).

    Returns:
        The reordered chunks; each ranked chunk gains ``scores["rerank"]``
        (the cross-encoder's relevance score in ``endpoint`` mode, reciprocal
        rank in ``llm`` mode).

    Raises:
        ValueError: If ``mode`` is not one of the three known modes — a typo
            must not silently select a mode that spends money.
    """
    if mode not in ("off", "llm", "endpoint"):
        raise ValueError(f"unknown rerank mode {mode!r} — expected 'off', 'llm' or 'endpoint'")
    if mode == "off" or len(chunks) <= 1:
        return chunks[:top_k] if top_k else chunks
    if mode == "endpoint" and endpoint_configured():
        ranked = _endpoint_rerank(query, chunks)
    else:
        ranked = _llm_rerank(query, chunks)
    return ranked[:top_k] if top_k else ranked


def _apply_order(chunks: list[ScoredChunk], order: list[int]) -> list[ScoredChunk]:
    """Apply a ranked index order defensively.

    Out-of-range and duplicate indices are dropped; candidates the reranker
    omitted keep their original relative order at the tail. Ranked chunks get
    ``scores["rerank"]`` = reciprocal rank — unless the caller already stamped
    a native score (``_endpoint_rerank`` stamps the cross-encoder's relevance
    score, which must survive)."""
    seen: set[int] = set()
    ranked: list[ScoredChunk] = []
    for idx in order:
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            c = chunks[idx]
            c.scores.setdefault("rerank", round(1.0 / (len(ranked) + 1), 4))
            ranked.append(c)
    for i, c in enumerate(chunks):
        if i not in seen:
            ranked.append(c)
    return ranked


def _llm_rerank(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """RankGPT-style listwise rerank: one numbered listing, one chat call.

    The listing shows each candidate's contextual text (blurb + chunk), not the
    bare chunk: an out-of-context chunk is ambiguous to the reranker for the
    same reason it was ambiguous to the retrievers, and would be re-shuffled
    instead of ranked. Same recipe as Anthropic's cookbook, which reranks over
    the chunk combined with its situating context."""
    listing = "\n".join(f"[{i}] {c.embed_text[:350]}" for i, c in enumerate(chunks))
    prompt = (
        "Rank the passages by how well they help answer the question. "
        f"Question: {query}\n\nPassages:\n{listing}\n\n"
        "Output ONLY the passage numbers from most to least relevant, comma-separated "
        "(e.g. 3,0,1). Most relevant first."
    )
    res = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    order = [int(x) for x in re.findall(r"\d+", res.text or "")]
    return _apply_order(chunks, order)


def _endpoint_rerank(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Call a hosted cross-encoder speaking the Cohere/Jina ``/rerank`` shape."""
    s = get_settings()
    body = json.dumps({
        "model": s.rerank_model, "query": query,
        # blurb + chunk, for the same reason as _llm_rerank
        "documents": [c.embed_text for c in chunks], "top_n": len(chunks),
    }).encode()
    url = s.rerank_base_url.rstrip("/") + "/rerank"
    headers = {"Content-Type": "application/json"}
    if s.rerank_api_key:
        headers["Authorization"] = f"Bearer {s.rerank_api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())
    results = sorted(data["results"], key=lambda r: r.get("relevance_score", 0), reverse=True)
    # Validate indices from the remote response BEFORE touching chunks: an
    # out-of-range or negative index must be dropped here, not crash the query
    # or stamp a score on the wrong chunk.
    order: list[int] = []
    for r in results:
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(chunks):
            chunks[idx].scores["rerank"] = round(r.get("relevance_score", 0), 4)
            order.append(idx)
    return _apply_order(chunks, order)
