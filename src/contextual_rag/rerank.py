"""Reranker seam — reorder fused candidates by true relevance to the query.

Three modes, all no-GPU:
  - "off"      : pass-through (the "optional, gracefully skipped" default).
  - "llm"      : RankGPT-style — one LLM call ranks all candidates. Works through the
                 existing chat endpoint, no extra service.
  - "endpoint" : a dedicated hosted cross-encoder reranker (Cohere/Jina shape), used
                 when RERANK_MODEL + RERANK_BASE_URL are configured.
"""

from __future__ import annotations

import json
import re
import urllib.request

from contextual_rag import llm
from contextual_rag.config import get_settings
from contextual_rag.types import ScoredChunk


def endpoint_configured() -> bool:
    s = get_settings()
    return bool(s.rerank_model and s.rerank_base_url)


def rerank(query: str, chunks: list[ScoredChunk], *, mode: str = "llm",
           top_k: int | None = None) -> list[ScoredChunk]:
    if mode == "off" or len(chunks) <= 1:
        return chunks[:top_k] if top_k else chunks
    if mode == "endpoint" and endpoint_configured():
        ranked = _endpoint_rerank(query, chunks)
    else:
        ranked = _llm_rerank(query, chunks)
    return ranked[:top_k] if top_k else ranked


def _apply_order(chunks: list[ScoredChunk], order: list[int]) -> list[ScoredChunk]:
    seen: set[int] = set()
    ranked: list[ScoredChunk] = []
    for idx in order:
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            c = chunks[idx]
            c.scores["rerank"] = round(1.0 / (len(ranked) + 1), 4)  # reciprocal rank
            ranked.append(c)
    for i, c in enumerate(chunks):  # keep any the reranker omitted, original order
        if i not in seen:
            ranked.append(c)
    return ranked


def _llm_rerank(query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    listing = "\n".join(f"[{i}] {c.text[:350]}" for i, c in enumerate(chunks))
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
    s = get_settings()
    body = json.dumps({
        "model": s.rerank_model, "query": query,
        "documents": [c.text for c in chunks], "top_n": len(chunks),
    }).encode()
    url = s.rerank_base_url.rstrip("/") + "/rerank"
    headers = {"Content-Type": "application/json"}
    if s.rerank_api_key:
        headers["Authorization"] = f"Bearer {s.rerank_api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())
    results = sorted(data["results"], key=lambda r: r.get("relevance_score", 0), reverse=True)
    for rank, r in enumerate(results):
        chunks[r["index"]].scores["rerank"] = round(r.get("relevance_score", 0), 4)
    return _apply_order(chunks, [r["index"] for r in results])
