"""Ingestion orchestrator — the one call that chains the steps into the memory:

    PDF / Markdown  ->  parse  ->  chunk  ->  contextualize  ->  embed + store

After this runs, the document is queryable via search()/answer().
"""

from __future__ import annotations

from contextual_rag.chunker import chunk_markdown, chunk_parsed_doc
from contextual_rag.contextual import contextualize_chunks
from contextual_rag.store import VectorStore


def ingest_parsed_doc(doc, *, domain_id: str = "default", store: VectorStore | None = None,
                      context_model: str | None = None) -> dict:
    """Ingest an already-parsed doc (chunk → contextualize → embed → store). Lets a
    caller reuse cached parser output instead of paying to re-parse a PDF."""
    chunks = chunk_parsed_doc(doc)
    ctx = contextualize_chunks(doc.markdown, chunks, model=context_model)
    store = store or VectorStore(domain_id)
    n = store.add(ctx)
    return {"doc_id": doc.filename, "pages": doc.pages, "chunks": n,
            "context_tokens": sum(c.total_tokens for c in ctx),
            "cached_tokens": sum(c.cached_tokens for c in ctx),
            "excerpted": any(c.excerpted for c in ctx)}


def ingest_pdf(data: bytes, filename: str, *, domain_id: str = "default",
               store: VectorStore | None = None, context_model: str | None = None) -> dict:
    from contextual_rag import parsing  # lazy: PDF support is an optional extra

    doc = parsing.parse_document(data, filename)
    return ingest_parsed_doc(doc, domain_id=domain_id, store=store, context_model=context_model)


def ingest_markdown(markdown: str, doc_id: str, *, domain_id: str = "default",
                    store: VectorStore | None = None, context_model: str | None = None) -> dict:
    chunks = chunk_markdown(markdown, doc_id=doc_id)
    ctx = contextualize_chunks(markdown, chunks, model=context_model)
    store = store or VectorStore(domain_id)
    n = store.add(ctx)
    return {"doc_id": doc_id, "chunks": n,
            "context_tokens": sum(c.total_tokens for c in ctx),
            "cached_tokens": sum(c.cached_tokens for c in ctx),
            "excerpted": any(c.excerpted for c in ctx)}
