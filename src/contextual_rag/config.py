"""Central configuration — the ONLY place environment variables are read.

The package talks to one kind of endpoint: any OpenAI-compatible API. The standard
OpenAI endpoint is the default; set OPENAI_BASE_URL to point at a compatible gateway
(an Azure-hosted proxy, LiteLLM, vLLM, a corporate gateway...). Switching endpoints
is a config change, never a code change.

Settings load lazily on first use, so importing the package never requires a key —
only actually calling an LLM does. Use `configure(...)` to set values from code.
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
    """Resolved configuration. `chat_model` / `embed_model` are whatever names the
    chosen endpoint expects — the rest of the package just passes them through."""

    # Endpoint. An empty base_url means the standard OpenAI API. The api_key may be
    # blank for keyless gateways (the client supplies a placeholder the server ignores).
    api_key: str = ""
    base_url: str = ""

    chat_model: str = "gpt-5.4-mini"
    embed_model: str = "text-embedding-3-large"
    # Vision model used to parse PDF pages (render → image → Markdown+LaTeX).
    # Should be a strong vision-capable model; falls back to chat_model if unset.
    parse_model: str = ""

    # Parser backend: "vision" (render → vision LLM, default, works anywhere the LLM
    # does) or "docintel" (Azure Document Intelligence — deterministic, in-tenant).
    parser: str = "vision"
    docintel_endpoint: str = ""
    docintel_key: str = ""

    # Where the embedded vector store persists (empty -> ./.contextual_rag).
    vector_dir: str = ""

    # Reranker seam (optional, no GPU). Empty rerank_model -> no dedicated reranker;
    # "llm" mode still works through the normal chat endpoint.
    rerank_model: str = ""
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    # Retrieval-time reranking of fused candidates: "llm" (RankGPT-style, one cheap
    # call through the chat endpoint), "endpoint" (the dedicated reranker above),
    # or "off". One of the biggest published retrieval wins.
    retrieval_rerank: str = "llm"

    # Contextualizer guard (see contextual.py). Documents at/under the cap are
    # situated against the WHOLE document (Anthropic's recipe verbatim); larger ones
    # against head+region excerpts of ~context_part_tokens. Keep the cap ≈20k tokens
    # under the chat model's usable input (250k fits the GPT-5 family; a 128k-context
    # model needs ~100k). Too high = hard API errors mid-ingestion.
    context_doc_cap: int = 250_000
    context_part_tokens: int = 48_000
    # Blurb calls per prompt-prefix group run 1 (cache-priming) + this many in
    # parallel. 1 = fully sequential; raise only within the key's rate-limit comfort.
    context_concurrency: int = 4

    # Generation default (overridable per call in llm.chat()).
    chat_temperature: float = 0.2


def _from_env() -> Settings:
    defaults = Settings()
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        chat_model=os.getenv("CHAT_MODEL", defaults.chat_model),
        embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
        parse_model=os.getenv("PARSE_MODEL", ""),
        parser=os.getenv("PARSER", defaults.parser).strip().lower(),
        docintel_endpoint=os.getenv("DOCINTEL_ENDPOINT", ""),
        docintel_key=os.getenv("DOCINTEL_KEY", ""),
        vector_dir=os.getenv("VECTOR_DIR", ""),
        rerank_model=os.getenv("RERANK_MODEL", ""),
        rerank_base_url=os.getenv("RERANK_BASE_URL", ""),
        rerank_api_key=os.getenv("RERANK_API_KEY", ""),
        retrieval_rerank=(os.getenv("RETRIEVAL_RERANK", "").strip().lower()
                          or defaults.retrieval_rerank),
        context_doc_cap=int(os.getenv("CONTEXT_DOC_CAP", str(defaults.context_doc_cap))),
        context_part_tokens=int(os.getenv("CONTEXT_PART_TOKENS",
                                          str(defaults.context_part_tokens))),
        context_concurrency=int(os.getenv("CONTEXT_CONCURRENCY",
                                          str(defaults.context_concurrency))),
        chat_temperature=float(os.getenv("CHAT_TEMPERATURE",
                                         str(defaults.chat_temperature))),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """The active settings, loaded from the environment on first use."""
    global _settings
    if _settings is None:
        _settings = _from_env()
    return _settings


def configure(**overrides) -> Settings:
    """Override settings from code (e.g. `configure(api_key=..., chat_model=...)`).
    Unknown keys raise; returns the new active settings."""
    global _settings
    _settings = replace(get_settings(), **overrides)
    return _settings


def reset_settings() -> None:
    """Drop overrides and reload from the environment on next use (mainly for tests)."""
    global _settings
    _settings = None
