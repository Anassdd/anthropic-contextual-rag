"""Vector store — persist contextualized chunks in Chroma and query them.

Chroma runs embedded (on-disk, no server, no GPU); one Chroma collection per
``domain_id`` keeps corpora isolated. Ingestion is incremental: ``upsert`` adds
or replaces documents without rebuilding anything.

What is stored per chunk (the contract that makes Contextual Retrieval work):

- **embedding** — vector of *blurb + chunk* (the blurb improves the match),
- **document** — the ORIGINAL chunk text (what gets shown to the LLM and cited),
- **metadata** — the blurb itself plus provenance (doc id, pages, section).

Embeddings go through the endpoint seam (:func:`contextual_rag.llm.embed`), so
the store never knows which model produced them — and never assumes their
dimension.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from contextual_rag import index_cache, llm
from contextual_rag.config import get_settings
from contextual_rag.types import ScoredChunk

_DEFAULT_DIR = ".contextual_rag"
_BATCH = 64  # embedding calls are batched to stay well under request-size limits


def _collection_name(domain_id: str) -> str:
    """Deterministic, Chroma-valid collection name for a domain.

    Chroma requires 3–512 chars of ``[a-zA-Z0-9._-]``, starting AND ending
    with an alphanumeric, with no ``".."``. Clean ids pass through unchanged
    (so existing stores keep their collections); whenever sanitization has to
    change the name — or the tail is not alphanumeric — a short hash of the
    ORIGINAL id is appended, so two distinct domain_ids can never silently
    collide into one collection (``"a b"`` vs ``"a_b"``).
    """
    raw = f"crag_{domain_id}"
    name = re.sub(r"\.\.+", "_", re.sub(r"[^a-zA-Z0-9._-]", "_", raw))
    if name != raw or not name[-1].isalnum():
        digest = hashlib.sha256(domain_id.encode()).hexdigest()[:8]
        name = f"{name[:494].rstrip('._-') or 'crag'}_{digest}"
    return name[:512]


def _embed(texts: list[str], model: str | None) -> list[list[float]]:
    """Embed ``texts`` in batches of ``_BATCH``, preserving order."""
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        out.extend(llm.embed(texts[i:i + _BATCH], model=model))
    return out


class VectorStore:
    """A persistent, per-collection vector store over contextualized chunks.

    Args:
        domain_id: Collection name — isolates corpora from each other.
        path: Directory where Chroma persists. Defaults to the ``VECTOR_DIR``
            setting, then ``./.contextual_rag``.

    Example:
        >>> store = VectorStore("my-corpus")
        >>> store.add(contextual_chunks)      # ContextualChunk objects
        >>> store.count()
        42
    """

    def __init__(self, domain_id: str = "default", *, path: str | None = None):
        import chromadb

        self.domain_id = domain_id
        # Resolved so two spellings of one directory ("./x", "x", "~/x") are
        # one corpus — and share one index-cache entry instead of two.
        raw = path or get_settings().vector_dir or _DEFAULT_DIR
        self._path = str(Path(raw).expanduser().resolve())
        #: Keys the query-side index cache (path + collection identify a corpus).
        self.cache_key = (self._path, domain_id)
        Path(self._path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._path)
        self._col = self._client.get_or_create_collection(
            _collection_name(domain_id), metadata={"hnsw:space": "cosine"})

    def __repr__(self) -> str:  # pragma: no cover - repr cosmetics
        return f"VectorStore(domain_id={self.domain_id!r}, path={self._path!r})"

    # ------------------------------------------------------------------ write

    def add(self, contextual_chunks, *, model: str | None = None) -> int:
        """Embed each chunk's contextual text and upsert it.

        Args:
            contextual_chunks: :class:`~contextual_rag.contextual.ContextualChunk`
                objects (the contextualizer's output).
            model: Override the configured embedding model.

        Returns:
            The number of chunks written.
        """
        items = list(contextual_chunks)
        if not items:
            return 0
        embeddings = _embed([c.text for c in items], model)
        self._col.upsert(
            ids=[c.chunk.chunk_id for c in items],
            embeddings=embeddings,
            documents=[c.chunk.text for c in items],  # original text (cited)
            metadatas=[{
                "context": c.context or "",
                "doc_id": c.chunk.doc_id,
                "pages": json.dumps(c.chunk.pages),
                "section": c.chunk.section,
                "domain_id": self.domain_id,
            } for c in items],
        )
        index_cache.invalidate(self.cache_key)
        return len(items)

    # ------------------------------------------------------------------- read

    def count(self) -> int:
        """Number of chunks stored in this collection."""
        return self._col.count()

    def doc_ids(self) -> set[str]:
        """Distinct doc_ids already stored.

        Each document is upserted in ONE :meth:`add` call, so presence means
        fully ingested — which lets bulk ingestion resume per document.
        """
        res = self._col.get(include=["metadatas"])
        return {m.get("doc_id") for m in res.get("metadatas") or [] if m and m.get("doc_id")}

    def _to_chunk(self, cid, doc, meta, score=0.0, scores=None) -> ScoredChunk:
        """Rehydrate one Chroma record into a ScoredChunk."""
        return ScoredChunk(
            chunk_id=cid, text=doc or "", context=meta.get("context", ""),
            doc_id=meta.get("doc_id", "?"),
            pages=json.loads(meta.get("pages", "[]") or "[]"),
            section=meta.get("section", ""), domain_id=meta.get("domain_id", self.domain_id),
            score=score, scores=scores or {})

    def get(self, chunk_id: str) -> ScoredChunk | None:
        """Fetch one chunk by id, or None if absent."""
        r = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not r["ids"]:
            return None
        return self._to_chunk(r["ids"][0], r["documents"][0], r["metadatas"][0])

    def query(self, query_embedding: list[float], k: int,
              *, doc_id: str | None = None) -> list[ScoredChunk]:
        """Dense search: cosine-nearest chunks to ``query_embedding``.

        Args:
            query_embedding: The query vector (same model as the stored chunks).
            k: Number of neighbours to return (capped at the collection size).
            doc_id: Restrict the search to one source document.

        Returns:
            Nearest chunks, best first, with ``scores["dense"]`` = cosine
            similarity.
        """
        n = min(k, self.count()) or 1
        where = {"doc_id": doc_id} if doc_id else None
        r = self._col.query(query_embeddings=[query_embedding], n_results=n, where=where,
                            include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                        r["metadatas"][0], r["distances"][0], strict=True):
            sim = round(1.0 - dist, 4)  # cosine distance -> similarity
            out.append(self._to_chunk(cid, doc, meta, score=sim, scores={"dense": sim}))
        return out

    def all_records(self) -> list[ScoredChunk]:
        """Every chunk in the collection (used to build the BM25 index)."""
        r = self._col.get(include=["documents", "metadatas"])
        return [self._to_chunk(cid, doc, meta)
                for cid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"], strict=True)]

    # ----------------------------------------------------------- housekeeping

    def reset(self) -> None:
        """Empty the collection but keep it usable."""
        self._client.delete_collection(_collection_name(self.domain_id))
        self._col = self._client.get_or_create_collection(
            _collection_name(self.domain_id), metadata={"hnsw:space": "cosine"})
        index_cache.invalidate(self.cache_key)

    def drop(self) -> None:
        """Delete the collection entirely (idempotent).

        Terminal: this instance must not be used afterwards — construct a new
        :class:`VectorStore` to recreate the collection."""
        try:
            self._client.delete_collection(_collection_name(self.domain_id))
        except Exception:
            pass
        index_cache.invalidate(self.cache_key)
