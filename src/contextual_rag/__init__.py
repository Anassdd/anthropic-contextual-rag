"""Contextual RAG — Anthropic's Contextual Retrieval technique as a package.

Pipeline: parse (optional) → chunk → contextualize → embed + BM25 → RRF →
rerank → grounded, source-cited answers. Works against any OpenAI-compatible
endpoint.

The one-object API::

    from contextual_rag import ContextualRAG

    rag = ContextualRAG("my-corpus")
    rag.ingest_markdown(text, doc_id="notes.md")   # or rag.ingest_pdf("doc.pdf")
    print(rag.ask("what does the corpus say about X?").text)

Every pipeline stage is also importable as a plain function — see the
``__all__`` groups below. Technique reference:
https://www.anthropic.com/news/contextual-retrieval
"""

from contextual_rag.answer import answer, answer_from
from contextual_rag.chunker import chunk_markdown, chunk_parsed_doc
from contextual_rag.config import ConfigError, Settings, configure, get_settings
from contextual_rag.contextual import ContextualChunk, contextualize_chunk, contextualize_chunks
from contextual_rag.ingest import ingest_markdown, ingest_parsed_doc, ingest_pdf
from contextual_rag.rag import ContextualRAG
from contextual_rag.search import rrf, search, search_trace
from contextual_rag.store import VectorStore
from contextual_rag.types import Answer, Chunk, RetrievalTrace, ScoredChunk

__version__ = "0.3.0"

__all__ = [
    # the facade
    "ContextualRAG",
    # pipeline steps, as functions
    "chunk_markdown", "chunk_parsed_doc",
    "contextualize_chunk", "contextualize_chunks", "ContextualChunk",
    "ingest_pdf", "ingest_markdown", "ingest_parsed_doc",
    "search", "search_trace", "rrf",
    "answer", "answer_from",
    # storage & types
    "VectorStore",
    "Chunk", "ScoredChunk", "RetrievalTrace", "Answer",
    # configuration
    "configure", "get_settings", "Settings", "ConfigError",
]
