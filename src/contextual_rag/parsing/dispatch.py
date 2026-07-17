"""Parser seam — pick the backend by config (mirrors the endpoint layer).

  PARSER=vision   (default) → render pages → vision LLM (works wherever the LLM does)
  PARSER=docintel           → Azure Document Intelligence (deterministic, in-tenant)

Both return a `ParsedDoc`, so callers don't care which ran. Switching backends is a
config change, no code change.
"""

from __future__ import annotations

from contextual_rag.config import get_settings
from contextual_rag.parsing import vision
from contextual_rag.parsing.base import ParsedDoc


def parse_document(data: bytes, filename: str, *, model: str | None = None) -> ParsedDoc:
    if get_settings().parser == "docintel":
        from contextual_rag.parsing import docintel  # lazy — only import the SDK path when chosen

        return docintel.parse(data, filename)  # DI is model-free; `model` applies to vision only
    return vision.parse_pdf(data, filename, model=model)
