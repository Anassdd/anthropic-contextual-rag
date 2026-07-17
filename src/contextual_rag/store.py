"""Vector store — embed contextualized chunks and persist them in Chroma (embedded,
on-disk, no server, no GPU). One collection per `domain_id`. Incremental: upsert adds
documents without rebuilding. Embeddings go through the endpoint layer (llm.embed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contextual_rag import index_cache, llm
from contextual_rag.config import get_settings
from contextual_rag.types import ScoredChunk

_DEFAULT_DIR = ".contextual_rag"
_BATCH = 64


def _collection_name(domain_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", f"crag_{domain_id}")
    return name[:512] if len(name) >= 3 else (name + "_x")


def _embed(texts: list[str], model: str | None) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        out.extend(llm.embed(texts[i:i + _BATCH], model=model))
    return out


class VectorStore:
    def __init__(self, domain_id: str = "default", *, path: str | None = None):
        import chromadb

        self.domain_id = domain_id
        self._path = path or get_settings().vector_dir or _DEFAULT_DIR
        self.cache_key = (self._path, domain_id)  # keys the query-side index cache
        Path(self._path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._path)
        self._col = self._client.get_or_create_collection(
            _collection_name(domain_id), metadata={"hnsw:space": "cosine"})

    # ---- write ----
    def add(self, contextual_chunks, *, model: str | None = None) -> int:
        """Embed each chunk's contextual text and upsert. `contextual_chunks` are
        ContextualChunk objects (the contextualizer's output)."""
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

    # ---- read ----
    def count(self) -> int:
        return self._col.count()

    def doc_ids(self) -> set[str]:
        """Distinct doc_ids already stored. Each doc is upserted in ONE add() call,
        so presence means fully ingested — lets bulk ingestion resume per document."""
        res = self._col.get(include=["metadatas"])
        return {m.get("doc_id") for m in res.get("metadatas") or [] if m and m.get("doc_id")}

    def _to_chunk(self, cid, doc, meta, score=0.0, scores=None) -> ScoredChunk:
        return ScoredChunk(
            chunk_id=cid, text=doc or "", context=meta.get("context", ""),
            doc_id=meta.get("doc_id", "?"),
            pages=json.loads(meta.get("pages", "[]") or "[]"),
            section=meta.get("section", ""), domain_id=meta.get("domain_id", self.domain_id),
            score=score, scores=scores or {})

    def get(self, chunk_id: str) -> ScoredChunk | None:
        r = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not r["ids"]:
            return None
        return self._to_chunk(r["ids"][0], r["documents"][0], r["metadatas"][0])

    def query(self, query_embedding: list[float], k: int,
              *, doc_id: str | None = None) -> list[ScoredChunk]:
        n = min(k, self.count()) or 1
        where = {"doc_id": doc_id} if doc_id else None
        r = self._col.query(query_embeddings=[query_embedding], n_results=n, where=where,
                            include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                        r["metadatas"][0], r["distances"][0]):
            sim = round(1.0 - dist, 4)  # cosine distance -> similarity
            out.append(self._to_chunk(cid, doc, meta, score=sim, scores={"dense": sim}))
        return out

    def all_records(self) -> list[ScoredChunk]:
        r = self._col.get(include=["documents", "metadatas"])
        return [self._to_chunk(cid, doc, meta)
                for cid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"])]

    def reset(self) -> None:
        self._client.delete_collection(_collection_name(self.domain_id))
        self._col = self._client.get_or_create_collection(
            _collection_name(self.domain_id), metadata={"hnsw:space": "cosine"})
        index_cache.invalidate(self.cache_key)

    def drop(self) -> None:
        """Delete this collection entirely."""
        try:
            self._client.delete_collection(_collection_name(self.domain_id))
        except Exception:
            pass
        index_cache.invalidate(self.cache_key)
