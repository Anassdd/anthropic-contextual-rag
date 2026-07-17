"""Contextual RAG — Anthropic-style Contextual Retrieval as a reusable package.

Pipeline: parse (optional) → chunk → contextualize → embed + BM25 → RRF → rerank →
grounded, source-cited answers. Works against any OpenAI-compatible endpoint.

    from contextual_rag import ContextualRAG
    rag = ContextualRAG("my-corpus")
    rag.ingest_markdown(text, doc_id="notes.md")
    print(rag.ask("what does the corpus say about X?").text)
"""

from contextual_rag.answer import answer, answer_from
from contextual_rag.chunker import chunk_markdown, chunk_parsed_doc
from contextual_rag.config import ConfigError, Settings, configure, get_settings
from contextual_rag.contextual import (ContextualChunk, contextualize_chunk,
                                       contextualize_chunks)
from contextual_rag.ingest import ingest_markdown, ingest_parsed_doc, ingest_pdf
from contextual_rag.rag import ContextualRAG
from contextual_rag.search import rrf, search, search_trace
from contextual_rag.store import VectorStore
from contextual_rag.types import Answer, Chunk, RetrievalTrace, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "ContextualRAG",
    # pipeline steps
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
