"""Parser seam — pick the backend by config (mirrors the endpoint seam).

- ``PARSER=vision`` (default) — render pages → vision LLM. Works wherever the
  LLM endpoint does; needs the ``pdf`` extra.
- ``PARSER=docintel`` — Azure Document Intelligence. Deterministic and fully
  in-tenant; needs the ``docintel`` extra.

Both return a :class:`~contextual_rag.parsing.base.ParsedDoc`, so callers
don't care which ran. Switching backends is a config change, no code change.
"""

from __future__ import annotations

from contextual_rag.config import get_settings
from contextual_rag.parsing import vision
from contextual_rag.parsing.base import ParsedDoc


def parse_document(data: bytes, filename: str, *, model: str | None = None) -> ParsedDoc:
    """Parse a PDF with the configured backend.

    Args:
        data: The PDF file bytes.
        filename: Carried into the result (becomes the doc_id downstream).
        model: Vision-model override; ignored by the model-free DI backend.

    Returns:
        The parsed document.

    Raises:
        ParseError: If the file can't be opened or the backend fails.
    """
    if get_settings().parser == "docintel":
        from contextual_rag.parsing import docintel  # lazy — only import the SDK path when chosen

        return docintel.parse(data, filename)
    return vision.parse_pdf(data, filename, model=model)
