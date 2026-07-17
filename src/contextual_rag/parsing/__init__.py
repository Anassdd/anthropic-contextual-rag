"""PDF parsing — turn a PDF into Markdown with page-level provenance.

Two interchangeable backends behind `parse_document`: the vision parser (default,
needs the `pdf` extra) and Azure Document Intelligence (needs the `docintel` extra).
"""

from contextual_rag.parsing.base import ParsedDoc, ParseError
from contextual_rag.parsing.dispatch import parse_document

__all__ = ["ParsedDoc", "ParseError", "parse_document"]
