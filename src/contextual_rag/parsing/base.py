"""Shared parser types — the common contract every backend returns.

Both the vision parser and the Azure DI parser return a `ParsedDoc`, so callers
(`parse_document`) never depend on which one ran. Lives here, apart from any one
backend, so neither backend has to import the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ParseError(ValueError):
    """A PDF could not be opened/rendered, or transcription failed."""


@dataclass
class ParsedDoc:
    filename: str
    pages: int  # pages actually processed
    total_pages: int
    page_markdown: list[str]
    markdown: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    routes: list[str] = field(default_factory=list)  # per page: "text" | "vision" | "docintel"

    @property
    def chars(self) -> int:
        return len(self.markdown)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def vision_pages(self) -> int:
        return self.routes.count("vision")

    @property
    def text_pages(self) -> int:
        return self.routes.count("text")
