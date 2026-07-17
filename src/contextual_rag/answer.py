"""Grounded answer — assemble retrieved chunks into a cited prompt and generate.

The generation is *constrained*: the model may only use the numbered sources it
is given, must cite them inline (``[S1]``, ``[S2]`` …), and must say so when
they don't contain the answer. Combined with the provenance every chunk
carries, each claim in an answer is verifiable against a document and page.
"""

from __future__ import annotations

from contextual_rag import llm
from contextual_rag.search import search
from contextual_rag.types import Answer, ScoredChunk

#: The grounding contract given to the model. Answering in the question's
#: language keeps multilingual corpora usable without translation plumbing.
SYSTEM = (
    "You are a precise assistant answering from a document library. Use ONLY the numbered "
    "sources below — do not use outside knowledge. Cite the sources you rely on inline as "
    "[S1], [S2], etc. If the sources do not contain the answer, say so plainly rather than "
    "guessing. Answer in the question's language."
)


def answer_from(query: str, chunks: list[ScoredChunk]) -> Answer:
    """Generate a grounded answer from an explicit set of source chunks.

    Args:
        query: The question, verbatim.
        chunks: The sources the answer is constrained to (typically
            ``search(...)`` output). ``[S{i+1}]`` in the answer refers to
            ``chunks[i]``.

    Returns:
        An :class:`~contextual_rag.types.Answer`. With no chunks at all, a
        fixed "no sources" answer is returned without calling the LLM.
    """
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
    """Retrieve top-k chunks for ``query`` and answer from them.

    Args:
        query: The question, verbatim.
        k: How many chunks to ground the answer on.
        domain_id: Collection to search (ignored when ``store`` is given).
        rerank_mode: ``"off"`` | ``"llm"`` | ``"endpoint"``.
        store: Reuse an existing :class:`~contextual_rag.store.VectorStore`.

    Returns:
        A grounded, ``[S#]``-cited :class:`~contextual_rag.types.Answer`.
    """
    return answer_from(query, search(query, k=k, domain_id=domain_id,
                                     rerank_mode=rerank_mode, store=store))
