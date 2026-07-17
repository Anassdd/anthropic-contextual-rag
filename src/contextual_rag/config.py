"""Central configuration — the only place environment variables are read.

The package talks to one kind of endpoint: any **OpenAI-compatible API**. The
standard OpenAI endpoint is the default; setting ``OPENAI_BASE_URL`` points the
whole pipeline at a compatible gateway instead (an Azure-hosted proxy, LiteLLM,
vLLM, a corporate gateway...). Switching endpoints is a config change, never a
code change.

Settings load **lazily** on first use, so importing the package never requires a
key — only actually calling an LLM does. Three ways to configure, in order of
precedence:

1. ``configure(...)`` from code (highest — overrides everything),
2. real environment variables,
3. a ``.env`` file in the working directory (lowest — never overrides real env).

Example:
    >>> from contextual_rag import configure, get_settings
    >>> configure(api_key="sk-...", chat_model="gpt-5.4-mini")
    >>> get_settings().chat_model
    'gpt-5.4-mini'
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv

# Picks up a ./.env in the consumer's project if present; real env vars win.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


@dataclass(frozen=True)
class Settings:
    """Resolved, immutable configuration for the whole pipeline.

    Model names are whatever the chosen endpoint expects — the rest of the
    package passes them through verbatim. That uniformity is what keeps
    :mod:`contextual_rag.llm` the single endpoint seam.

    Attributes:
        api_key: API key. May be blank for keyless gateways (the client then
            sends a placeholder the server ignores).
        base_url: Custom OpenAI-compatible endpoint. Empty = standard OpenAI.
        chat_model: Model used for situating blurbs, LLM reranking and answers.
        embed_model: Embedding model. Pick a multilingual-strong one for
            non-English corpora; the vector dimension is read dynamically.
        parse_model: Vision model for PDF page transcription. Empty = falls
            back to ``chat_model``.
        parser: PDF parser backend, ``"vision"`` or ``"docintel"``.
        docintel_endpoint: Azure Document Intelligence endpoint (docintel only).
        docintel_key: Azure Document Intelligence key (docintel only).
        vector_dir: Where the vector store persists. Empty = ``./.contextual_rag``.
        rerank_model: Model name for a dedicated hosted reranker (endpoint mode).
        rerank_base_url: Base URL of the hosted reranker.
        rerank_api_key: Key for the hosted reranker (optional).
        retrieval_rerank: Default rerank mode at query time:
            ``"llm"`` | ``"endpoint"`` | ``"off"``.
        context_doc_cap: Documents at/under this many tokens are situated
            against the whole document (Anthropic's recipe verbatim); larger
            ones switch to excerpt mode. Keep ≈20k tokens under the chat
            model's usable input — 250k fits the GPT-5 family, a 128k-context
            model needs ~100k. Too high = hard API errors mid-ingestion.
        context_part_tokens: Target size of one excerpt in excerpt mode.
        context_concurrency: Parallel blurb calls per prompt-prefix group
            (1 = fully sequential; raise within the key's rate limits).
        chat_temperature: Default generation temperature (overridable per call).
    """

    api_key: str = ""
    base_url: str = ""

    chat_model: str = "gpt-5.4-mini"
    embed_model: str = "text-embedding-3-large"
    parse_model: str = ""

    parser: str = "vision"
    docintel_endpoint: str = ""
    docintel_key: str = ""

    vector_dir: str = ""

    rerank_model: str = ""
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    retrieval_rerank: str = "llm"

    context_doc_cap: int = 250_000
    context_part_tokens: int = 48_000
    context_concurrency: int = 4

    chat_temperature: float = 0.2


def _from_env() -> Settings:
    """Build a :class:`Settings` from the environment, defaulting field by field."""
    d = Settings()  # source of default values — never duplicated here
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        chat_model=os.getenv("CHAT_MODEL", d.chat_model),
        embed_model=os.getenv("EMBED_MODEL", d.embed_model),
        parse_model=os.getenv("PARSE_MODEL", ""),
        parser=os.getenv("PARSER", d.parser).strip().lower(),
        docintel_endpoint=os.getenv("DOCINTEL_ENDPOINT", ""),
        docintel_key=os.getenv("DOCINTEL_KEY", ""),
        vector_dir=os.getenv("VECTOR_DIR", ""),
        rerank_model=os.getenv("RERANK_MODEL", ""),
        rerank_base_url=os.getenv("RERANK_BASE_URL", ""),
        rerank_api_key=os.getenv("RERANK_API_KEY", ""),
        retrieval_rerank=(os.getenv("RETRIEVAL_RERANK", "").strip().lower()
                          or d.retrieval_rerank),
        context_doc_cap=int(os.getenv("CONTEXT_DOC_CAP", str(d.context_doc_cap))),
        context_part_tokens=int(os.getenv("CONTEXT_PART_TOKENS",
                                          str(d.context_part_tokens))),
        context_concurrency=int(os.getenv("CONTEXT_CONCURRENCY",
                                          str(d.context_concurrency))),
        chat_temperature=float(os.getenv("CHAT_TEMPERATURE", str(d.chat_temperature))),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the active settings, loading them from the environment on first use.

    Returns:
        The process-wide :class:`Settings` instance. The same object is returned
        until :func:`configure` or :func:`reset_settings` replaces it — modules
        that cache per-settings state (the LLM client) key off its identity.
    """
    global _settings
    if _settings is None:
        _settings = _from_env()
    return _settings


def configure(**overrides) -> Settings:
    """Override settings from code, e.g. ``configure(api_key="sk-...")``.

    Values not overridden keep their current value (env-loaded or previously
    configured). The LLM client is rebuilt on the next call automatically.

    Args:
        **overrides: Any :class:`Settings` field name → new value.

    Returns:
        The new active :class:`Settings`.

    Raises:
        TypeError: If an override is not a :class:`Settings` field.
    """
    global _settings
    _settings = replace(get_settings(), **overrides)
    return _settings


def reset_settings() -> None:
    """Drop all overrides; the next :func:`get_settings` reloads from the
    environment. Mainly useful in tests."""
    global _settings
    _settings = None
