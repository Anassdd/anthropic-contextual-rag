"""Configuration tests — lazy loading, precedence, and loud failure on typos.

Misconfiguration must fail with a message naming the setting, never silently
pick a behavior that spends money (RETRIEVAL_RERANK=none silently meaning
"llm" was a real hazard). Each test isolates the environment: the module-level
settings cache is reset and .env loading is disabled.
"""

import pytest

from contextual_rag import config
from contextual_rag.config import (
    ConfigError,
    Settings,
    configure,
    get_settings,
    reset_settings,
)

_VARS = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_MODEL", "EMBED_MODEL",
    "PARSE_MODEL", "PARSER", "VECTOR_DIR", "RETRIEVAL_RERANK",
    "RERANK_MODEL", "RERANK_BASE_URL", "RERANK_API_KEY",
    "CONTEXT_DOC_CAP", "CONTEXT_PART_TOKENS", "CONTEXT_CONCURRENCY",
    "CHAT_TEMPERATURE", "DOCINTEL_ENDPOINT", "DOCINTEL_KEY",
]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_defaults_load_with_zero_configuration():
    s = get_settings()
    assert s.retrieval_rerank == "llm" and s.parser == "vision"
    assert s.context_doc_cap == 250_000


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "my-model")
    monkeypatch.setenv("CONTEXT_CONCURRENCY", "9")
    s = get_settings()
    assert s.chat_model == "my-model" and s.context_concurrency == 9


def test_configure_beats_env(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "env-model")
    configure(chat_model="code-model")
    assert get_settings().chat_model == "code-model"


def test_enum_fields_are_normalized():
    s = Settings(retrieval_rerank=" LLM ", parser="Vision")
    assert s.retrieval_rerank == "llm" and s.parser == "vision"


def test_rerank_typo_fails_loudly(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_RERANK", "none")  # plausible typo for "off"
    with pytest.raises(ConfigError, match="retrieval_rerank"):
        get_settings()


def test_parser_typo_fails_loudly():
    with pytest.raises(ConfigError, match="parser"):
        configure(parser="visual")


def test_malformed_numeric_env_names_the_variable(monkeypatch):
    monkeypatch.setenv("CONTEXT_DOC_CAP", "a-lot")
    with pytest.raises(ConfigError, match="CONTEXT_DOC_CAP"):
        get_settings()


def test_empty_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CONTEXT_DOC_CAP", "")
    assert get_settings().context_doc_cap == 250_000


def test_failed_configure_leaves_settings_intact():
    configure(chat_model="keep-me")
    with pytest.raises(ConfigError):
        configure(retrieval_rerank="bogus")
    assert get_settings().chat_model == "keep-me", \
        "a rejected configure() must not corrupt the active settings"
