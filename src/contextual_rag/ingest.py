"""Ingestion orchestrator — one call from raw document to queryable memory::

    PDF / Markdown ─▶ parse ─▶ chunk ─▶ contextualize ─▶ embed ─▶ store

After any ``ingest_*`` call returns, the document is queryable via
``search()`` / ``answer()``.

Every function returns an **ingestion report** dict:

- ``doc_id`` — the id the chunks were stored under,
- ``chunks`` — how many chunks were written,
- ``context_tokens`` — total tokens the contextualizer spent,
- ``cached_tokens`` — how many of those were served from the prompt-prefix
  cache (verify caching works: expect a large fraction on multi-chunk docs),
- ``excerpted`` — True if the document was too large for whole-document mode,
- ``pages`` — page count (PDF ingestion only).
"""

from __future__ import annotations

from contextual_rag.chunker import chunk_markdown, chunk_parsed_doc
from contextual_rag.contextual import contextualize_chunks
from contextual_rag.store import VectorStore


def ingest_parsed_doc(doc, *, domain_id: str = "default", store: VectorStore | None = None,
                      context_model: str | None = None) -> dict:
    """Ingest an already-parsed document (chunk → contextualize → embed → store).

    Lets a caller reuse cached parser output instead of paying to re-parse.

    Args:
        doc: A :class:`~contextual_rag.parsing.ParsedDoc` (or anything with
            ``markdown``, ``page_markdown``, ``filename`` and ``pages``).
        domain_id: Collection to store into (ignored when ``store`` is given).
        store: Reuse an existing :class:`VectorStore`.
        context_model: Override the blurb model (a cheap tier is fine).

    Returns:
        The ingestion report (see module docstring).
    """
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
    """Ingest a PDF: parse (vision or Document Intelligence), then the pipeline.

    Requires the ``pdf`` extra (vision parser) or ``docintel`` extra.

    Args:
        data: The PDF file bytes.
        filename: Stored as the ``doc_id`` (and used in citations).
        domain_id: Collection to store into (ignored when ``store`` is given).
        store: Reuse an existing :class:`VectorStore`.
        context_model: Override the blurb model.

    Returns:
        The ingestion report (see module docstring).
    """
    from contextual_rag import parsing  # lazy: PDF support is an optional extra

    doc = parsing.parse_document(data, filename)
    return ingest_parsed_doc(doc, domain_id=domain_id, store=store, context_model=context_model)


def ingest_markdown(markdown: str, doc_id: str, *, domain_id: str = "default",
                    store: VectorStore | None = None, context_model: str | None = None) -> dict:
    """Ingest Markdown (or plain) text directly — no parsing step.

    Args:
        markdown: The document text.
        doc_id: Stored as the document id (and used in citations).
        domain_id: Collection to store into (ignored when ``store`` is given).
        store: Reuse an existing :class:`VectorStore`.
        context_model: Override the blurb model.

    Returns:
        The ingestion report (see module docstring).
    """
    chunks = chunk_markdown(markdown, doc_id=doc_id)
    ctx = contextualize_chunks(markdown, chunks, model=context_model)
    store = store or VectorStore(domain_id)
    n = store.add(ctx)
    return {"doc_id": doc_id, "chunks": n,
            "context_tokens": sum(c.total_tokens for c in ctx),
            "cached_tokens": sum(c.cached_tokens for c in ctx),
            "excerpted": any(c.excerpted for c in ctx)}
