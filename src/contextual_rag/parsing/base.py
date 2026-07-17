"""Shared parser types — the common contract every backend returns.

Both the vision parser and the Azure Document Intelligence parser return a
:class:`ParsedDoc`, so callers (:func:`~contextual_rag.parsing.parse_document`)
never depend on which one ran. Lives apart from any one backend so neither
backend has to import the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ParseError(ValueError):
    """A PDF could not be opened/rendered, or transcription failed."""


@dataclass
class ParsedDoc:
    """A parsed document: Markdown text plus page provenance and cost data.

    Attributes:
        filename: The source file's name (becomes the ``doc_id`` downstream).
        pages: Pages actually processed (≤ ``total_pages`` when capped).
        total_pages: Pages in the source file.
        page_markdown: Per-page Markdown — what page provenance derives from.
        markdown: The whole document as one Markdown string.
        model: The vision model that transcribed (or a backend marker).
        prompt_tokens: Vision-call prompt tokens spent.
        completion_tokens: Vision-call completion tokens spent.
        routes: Per page, which path handled it: ``"text"`` (free text layer),
            ``"vision"`` (rendered + transcribed) or ``"docintel"``.
    """

    filename: str
    pages: int
    total_pages: int
    page_markdown: list[str]
    markdown: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    routes: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        """Length of the full Markdown, in characters."""
        return len(self.markdown)

    @property
    def total_tokens(self) -> int:
        """Total vision tokens the parse cost."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def vision_pages(self) -> int:
        """How many pages needed a vision call."""
        return self.routes.count("vision")

    @property
    def text_pages(self) -> int:
        """How many pages rode the free text layer."""
        return self.routes.count("text")
