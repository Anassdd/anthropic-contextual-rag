"""Shared types — the stable contracts the pipeline stages exchange.

:class:`Chunk` is what the chunker produces; :class:`ScoredChunk` is what
retrieval returns; :func:`~contextual_rag.answer.answer_from` depends only on
:class:`ScoredChunk`. Any retriever honouring that contract — a knowledge
graph, a SQL retriever — can slot in next to the built-in ones without a
rewrite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """A retrievable passage plus the provenance needed to cite it.

    Attributes:
        chunk_id: Stable id, ``"<doc_id>::<index>"``.
        doc_id: The source document's identifier (usually its filename).
        index: 0-based position of the chunk within its document.
        text: The chunk's content — every character of the source survives
            chunking, so nothing is lost or un-citable.
        header_path: Heading trail down to this chunk, e.g.
            ``["Methods", "Data"]``.
        pages: Source page numbers this chunk spans (page provenance).
        token_count: Size in tokens (per the counter used at chunking time).
        char_count: Size in characters.
        overlap_tokens: Leading tokens carried over from the previous chunk.
        domain_id: The collection this chunk belongs to.
    """

    chunk_id: str
    doc_id: str
    index: int
    text: str
    header_path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    token_count: int = 0
    char_count: int = 0
    overlap_tokens: int = 0
    domain_id: str = "default"

    @property
    def section(self) -> str:
        """The heading trail as one string, e.g. ``"Methods › Data"``."""
        return " › ".join(self.header_path)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (adds the computed ``section``)."""
        d = asdict(self)
        d["section"] = self.section
        return d


@dataclass
class ScoredChunk:
    """A retrieved chunk with its per-stage scores and citation metadata.

    The split between ``text`` and ``context`` is the heart of Contextual
    Retrieval: the situating blurb (``context``) improves *findability* — it is
    part of what gets embedded and BM25-indexed — but the *original* chunk text
    is what gets shown to the LLM and cited to the user.

    Attributes:
        chunk_id: Stable id, ``"<doc_id>::<index>"``.
        text: The ORIGINAL chunk text — fed to the LLM, cited to the user.
        context: The situating blurb — indexed, never cited.
        doc_id: Source document id.
        pages: Source page numbers (provenance).
        section: Heading trail, e.g. ``"Methods › Data"``.
        domain_id: The collection this chunk belongs to.
        score: The chunk's current overall score (RRF after fusion).
        scores: Per-stage scores: ``dense`` / ``bm25`` / ``rrf`` / ``rerank``.
    """

    chunk_id: str
    text: str
    context: str
    doc_id: str
    pages: list[int]
    section: str
    domain_id: str = "default"
    score: float = 0.0
    scores: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable provenance, e.g. ``"report.pdf, p.12-13"``."""
        pages = ", p." + "-".join(str(p) for p in self.pages) if self.pages else ""
        return f"{self.doc_id}{pages}"

    @property
    def embed_text(self) -> str:
        """What gets embedded / BM25-indexed: the blurb prepended to the chunk."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


@dataclass
class RetrievalTrace:
    """Every stage a query passes through — for debugging and step-by-step UIs.

    Attributes:
        query: The query as asked.
        dense: Embedding-similarity ranking (best first).
        bm25: Keyword (BM25) ranking.
        fused: After Reciprocal Rank Fusion of the two rankings.
        reranked: After the (optional) reranker; equals ``fused`` if skipped.
        final: The top-k actually returned to the caller.
        reranked_applied: Whether a reranker actually ran.
        timings: Per-stage wall-clock in ms: ``dense_ms``/``bm25_ms``/``rerank_ms``.
    """

    query: str
    dense: list[ScoredChunk] = field(default_factory=list)
    bm25: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    reranked: list[ScoredChunk] = field(default_factory=list)
    final: list[ScoredChunk] = field(default_factory=list)
    reranked_applied: bool = False
    timings: dict = field(default_factory=dict)


@dataclass
class Answer:
    """A grounded answer plus the sources it was constrained to.

    Attributes:
        text: The generated answer, with inline ``[S1]``-style citations.
        sources: The chunks the answer was allowed to use; ``[S{i+1}]`` in the
            text refers to ``sources[i]``.
        prompt_tokens: Tokens spent on the generation prompt.
        completion_tokens: Tokens generated.
    """

    text: str
    sources: list[ScoredChunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0
