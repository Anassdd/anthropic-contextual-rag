"""Shared types — the stable contracts the pipeline stages exchange.

`Chunk` is what the chunker produces; `ScoredChunk` is what retrieval returns;
`answer()` depends only on `ScoredChunk`, so any retriever honouring that contract
can slot in without a rewrite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """A retrievable passage plus the provenance needed to cite it (doc + page +
    section) and to rebuild its surrounding context."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    header_path: list[str] = field(default_factory=list)  # ["Methods", "Data"]
    pages: list[int] = field(default_factory=list)  # source page numbers (provenance)
    token_count: int = 0
    char_count: int = 0
    overlap_tokens: int = 0  # leading tokens carried over from the previous chunk
    domain_id: str = "default"

    @property
    def section(self) -> str:
        return " › ".join(self.header_path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["section"] = self.section
        return d


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str           # the ORIGINAL chunk text — what we feed the LLM and cite
    context: str        # the contextual blurb (used for embedding/BM25, not cited)
    doc_id: str
    pages: list[int]
    section: str
    domain_id: str = "default"
    score: float = 0.0
    scores: dict = field(default_factory=dict)  # per-stage: dense / bm25 / rrf / rerank

    @property
    def citation(self) -> str:
        pages = ", p." + "-".join(str(p) for p in self.pages) if self.pages else ""
        return f"{self.doc_id}{pages}"

    @property
    def embed_text(self) -> str:
        """What gets embedded / BM25-indexed: the blurb prepended to the chunk."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


@dataclass
class RetrievalTrace:
    """Every stage a query passes through — for debugging and step-by-step UIs."""

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
    text: str
    sources: list[ScoredChunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0
