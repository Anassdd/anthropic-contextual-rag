"""ContextualRAG — the one-object facade over the whole pipeline.

    rag = ContextualRAG("my-corpus")
    rag.ingest_markdown(text, doc_id="notes.md")     # or rag.ingest_pdf("report.pdf")
    hits = rag.search("what does the report say about X?")
    ans = rag.ask("what does the report say about X?")   # grounded, [S#]-cited

Everything the facade does is also available as plain functions (ingest_*, search,
answer) for callers who prefer to manage the `VectorStore` themselves.
"""

from __future__ import annotations

from pathlib import Path

from contextual_rag import answer as _answer
from contextual_rag import ingest as _ingest
from contextual_rag import search as _search
from contextual_rag.config import get_settings
from contextual_rag.store import VectorStore
from contextual_rag.types import Answer, RetrievalTrace, ScoredChunk


class ContextualRAG:
    """One retrieval corpus: a named Chroma collection plus the pipeline around it.

    `collection` isolates corpora from each other; `path` overrides where the store
    persists (default: VECTOR_DIR or ./.contextual_rag). `rerank` fixes the rerank
    mode ("llm" | "endpoint" | "off"); None defers to RETRIEVAL_RERANK at query time.
    """

    def __init__(self, collection: str = "default", *, path: str | None = None,
                 rerank: str | None = None):
        self.store = VectorStore(collection, path=path)
        self._rerank = rerank

    @property
    def rerank_mode(self) -> str:
        return self._rerank if self._rerank is not None else get_settings().retrieval_rerank

    # ---- ingest ----
    def ingest_pdf(self, source: bytes | str | Path, filename: str | None = None,
                   *, context_model: str | None = None) -> dict:
        """Ingest a PDF from bytes or a file path (parse → chunk → contextualize →
        embed → store). Returns an ingestion report dict."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            data = path.read_bytes()
            filename = filename or path.name
        else:
            data = source
            filename = filename or "document.pdf"
        return _ingest.ingest_pdf(data, filename, store=self.store,
                                  context_model=context_model)

    def ingest_markdown(self, markdown: str, doc_id: str,
                        *, context_model: str | None = None) -> dict:
        """Ingest Markdown (or plain) text directly — no parsing step."""
        return _ingest.ingest_markdown(markdown, doc_id, store=self.store,
                                       context_model=context_model)

    def ingest_parsed_doc(self, doc, *, context_model: str | None = None) -> dict:
        """Ingest an already-parsed `ParsedDoc` (reuse cached parser output)."""
        return _ingest.ingest_parsed_doc(doc, store=self.store,
                                         context_model=context_model)

    # ---- query ----
    def search(self, query: str, *, k: int = 8, doc_id: str | None = None) -> list[ScoredChunk]:
        return _search.search(query, k=k, store=self.store,
                              rerank_mode=self.rerank_mode, doc_id=doc_id)

    def search_trace(self, query: str, *, k: int = 8,
                     doc_id: str | None = None) -> RetrievalTrace:
        """Like search(), but returns every stage (dense/bm25/fused/reranked/final)."""
        return _search.search_trace(query, k=k, store=self.store,
                                    rerank_mode=self.rerank_mode, doc_id=doc_id)

    def ask(self, query: str, *, k: int = 6) -> Answer:
        """Retrieve then generate a grounded answer with [S#] citations."""
        return _answer.answer_from(query, self.search(query, k=k))

    # ---- housekeeping ----
    def count(self) -> int:
        return self.store.count()

    def doc_ids(self) -> set[str]:
        return self.store.doc_ids()

    def reset(self) -> None:
        """Empty the collection (keeps it usable)."""
        self.store.reset()
