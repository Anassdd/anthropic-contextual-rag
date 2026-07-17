"""Grounded answer — assemble retrieved chunks into a source-cited prompt and generate.

The answer is constrained to the retrieved sources and must cite them, so it stays
grounded and the user can verify against doc + page (provenance).
"""

from __future__ import annotations

from contextual_rag import llm
from contextual_rag.search import search
from contextual_rag.types import Answer, ScoredChunk

SYSTEM = (
    "You are a precise assistant answering from a document library. Use ONLY the numbered "
    "sources below — do not use outside knowledge. Cite the sources you rely on inline as "
    "[S1], [S2], etc. If the sources do not contain the answer, say so plainly rather than "
    "guessing. Answer in the question's language."
)


def answer_from(query: str, chunks: list[ScoredChunk]) -> Answer:
    if not chunks:
        return Answer(text="No indexed sources are available to answer from.", sources=[])
    block = "\n\n".join(
        f"[S{i + 1}] (source: {c.citation})\n{c.text}" for i, c in enumerate(chunks))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n{block}\n\nQuestion: {query}"},
    ]
    res = llm.chat(messages, temperature=0.0)
    u = res.usage
    return Answer(text=res.text or "", sources=chunks,
                  prompt_tokens=u.prompt_tokens if u else 0,
                  completion_tokens=u.completion_tokens if u else 0)


def answer(query: str, *, k: int = 6, domain_id: str = "default",
           rerank_mode: str = "off", store=None) -> Answer:
    return answer_from(query, search(query, k=k, domain_id=domain_id,
                                     rerank_mode=rerank_mode, store=store))
