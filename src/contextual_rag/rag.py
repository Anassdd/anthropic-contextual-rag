"""ContextualRAG — the one-object facade over the whole pipeline.

For the common case, this class is the entire API::

    from contextual_rag import ContextualRAG

    rag = ContextualRAG("my-corpus")
    rag.ingest_pdf("report.pdf")                       # or ingest_markdown(...)
    hits = rag.search("What drove Q3 margins?")        # ranked ScoredChunks
    ans = rag.ask("What drove Q3 margins?")            # grounded, [S#]-cited
    print(ans.text, [s.citation for s in ans.sources])

Everything the facade does is also available as plain functions
(``ingest_*``, ``search``, ``answer``) for callers who prefer to manage the
:class:`~contextual_rag.store.VectorStore` themselves.
"""

from __future__ import annotations

from pathlib import Path

# Functions are imported straight from their submodules — never through the
# package (`from contextual_rag import answer`): the package __init__ re-exports
# functions named `answer` and `search` that shadow those submodules as package
# attributes, so a package-level lookup resolves to module or function depending
# on import order. Direct submodule imports are immune to that.
from contextual_rag.answer import answer_from as _answer_from
from contextual_rag.config import get_settings
from contextual_rag.ingest import ingest_markdown as _ingest_markdown
from contextual_rag.ingest import ingest_parsed_doc as _ingest_parsed_doc
from contextual_rag.ingest import ingest_pdf as _ingest_pdf
from contextual_rag.search import search as _search
from contextual_rag.search import search_trace as _search_trace
from contextual_rag.store import VectorStore
from contextual_rag.types import Answer, RetrievalTrace, ScoredChunk


class ContextualRAG:
    """One retrieval corpus: a named, persistent collection plus the pipeline.

    Args:
        collection: Collection name — isolates corpora from each other. The
            same name on the same ``path`` reopens the same corpus.
        path: Where the store persists. Defaults to the ``VECTOR_DIR``
            setting, then ``./.contextual_rag``.
        rerank: Fix the rerank mode for this instance
            (``"llm"`` | ``"endpoint"`` | ``"off"``). None (default) defers to
            the ``RETRIEVAL_RERANK`` setting, resolved at query time.

    Example:
        >>> rag = ContextualRAG("docs")
        >>> rag.ingest_markdown("# Note\\n\\nRRF fuses rankings.", doc_id="note.md")
        >>> rag.ask("What does RRF do?").text
        'RRF fuses rankings ... [S1]'
    """

    def __init__(self, collection: str = "default", *, path: str | None = None,
                 rerank: str | None = None):
        self.store = VectorStore(collection, path=path)
        self._rerank = rerank

    def __repr__(self) -> str:  # pragma: no cover - repr cosmetics
        return (f"ContextualRAG(collection={self.store.domain_id!r}, "
                f"chunks={self.count()}, rerank={self.rerank_mode!r})")

    @property
    def rerank_mode(self) -> str:
        """The rerank mode in effect (instance override, else the setting)."""
        return self._rerank if self._rerank is not None else get_settings().retrieval_rerank

    # ----------------------------------------------------------------- ingest

    def ingest_pdf(self, source: bytes | str | Path, filename: str | None = None,
                   *, context_model: str | None = None) -> dict:
        """Ingest a PDF from bytes or a file path.

        Runs parse → chunk → contextualize → embed → store. Requires the
        ``pdf`` extra (vision parser) or ``docintel`` extra.

        Args:
            source: PDF bytes, or a path to a PDF file.
            filename: Overrides the stored doc_id (defaults to the file's
                name; required to be meaningful when passing bytes).
            context_model: Override the blurb model.

        Returns:
            The ingestion report (see :mod:`contextual_rag.ingest`).
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            data = path.read_bytes()
            filename = filename or path.name
        else:
            data = source
            filename = filename or "document.pdf"
        return _ingest_pdf(data, filename, store=self.store,
                           context_model=context_model)

    def ingest_markdown(self, markdown: str, doc_id: str,
                        *, context_model: str | None = None) -> dict:
        """Ingest Markdown (or plain) text directly — no parsing step.

        Args:
            markdown: The document text.
            doc_id: Stored as the document id (and used in citations).
            context_model: Override the blurb model.

        Returns:
            The ingestion report (see :mod:`contextual_rag.ingest`).
        """
        return _ingest_markdown(markdown, doc_id, store=self.store,
                                context_model=context_model)

    def ingest_parsed_doc(self, doc, *, context_model: str | None = None) -> dict:
        """Ingest an already-parsed ``ParsedDoc`` (reuse cached parser output)."""
        return _ingest_parsed_doc(doc, store=self.store,
                                  context_model=context_model)

    # ------------------------------------------------------------------ query

    def search(self, query: str, *, k: int = 8,
               doc_id: str | None = None) -> list[ScoredChunk]:
        """Hybrid search over the corpus.

        Args:
            query: The question, verbatim.
            k: Number of chunks to return.
            doc_id: Scope the search to one source document.

        Returns:
            The top-k chunks, best first.
        """
        return _search(query, k=k, store=self.store,
                       rerank_mode=self.rerank_mode, doc_id=doc_id)

    def search_trace(self, query: str, *, k: int = 8,
                     doc_id: str | None = None) -> RetrievalTrace:
        """Like :meth:`search`, but returns every retrieval stage
        (dense / bm25 / fused / reranked / final) with scores and timings."""
        return _search_trace(query, k=k, store=self.store,
                             rerank_mode=self.rerank_mode, doc_id=doc_id)

    def ask(self, query: str, *, k: int = 6) -> Answer:
        """Retrieve, then generate a grounded answer with ``[S#]`` citations.

        Args:
            query: The question, verbatim.
            k: How many chunks to ground the answer on.

        Returns:
            An :class:`~contextual_rag.types.Answer`; each ``[S{i+1}]`` in
            ``.text`` refers to ``.sources[i]``.
        """
        return _answer_from(query, self.search(query, k=k))

    # ----------------------------------------------------------- housekeeping

    def count(self) -> int:
        """Number of chunks in the corpus."""
        return self.store.count()

    def doc_ids(self) -> set[str]:
        """Doc ids already ingested (useful to skip re-ingestion)."""
        return self.store.doc_ids()

    def reset(self) -> None:
        """Empty the corpus (keeps the collection usable)."""
        self.store.reset()
