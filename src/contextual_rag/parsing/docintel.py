"""Azure Document Intelligence parser — a deterministic alternative backend.

Sends the PDF to Azure DI's `prebuilt-layout` model and returns Markdown (tables as
HTML) + formulas as LaTeX. Deterministic — it doesn't hallucinate — and fully
in-tenant, which matters for confidential corpora.

Provenance caveat: DI returns the document as ONE unified Markdown string, and
mapping its per-element page spans onto that string is not wired up yet — so
chunks from this backend carry no page numbers (citations name the document
only). The vision backend provides real per-page provenance.

Selected by `PARSER=docintel`; needs `DOCINTEL_ENDPOINT` + `DOCINTEL_KEY` and the
`docintel` extra installed.
"""

from __future__ import annotations

from contextual_rag.config import get_settings
from contextual_rag.parsing.base import ParsedDoc, ParseError


def parse(data: bytes, filename: str) -> ParsedDoc:
    settings = get_settings()
    if not settings.docintel_endpoint or not settings.docintel_key:
        raise ParseError(
            "Document Intelligence not configured — set DOCINTEL_ENDPOINT and "
            "DOCINTEL_KEY (and PARSER=docintel)."
        )
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import (
            AnalyzeDocumentRequest,
            DocumentAnalysisFeature,
            DocumentContentFormat,
        )
        from azure.core.credentials import AzureKeyCredential
    except Exception as exc:  # pragma: no cover
        raise ParseError(f"azure-ai-documentintelligence not installed: {exc}") from exc

    client = DocumentIntelligenceClient(
        settings.docintel_endpoint, AzureKeyCredential(settings.docintel_key)
    )
    try:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=data),
            features=[DocumentAnalysisFeature.FORMULAS],  # equations -> LaTeX (paid add-on)
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()
    except Exception as exc:
        raise ParseError(f"Document Intelligence request failed: {exc}") from exc

    markdown = (result.content or "").strip()
    if not markdown:
        raise ParseError("Document Intelligence returned no content.")
    pages = len(result.pages or []) or 1
    return ParsedDoc(
        filename=filename,
        pages=pages,
        total_pages=pages,
        # One unified blob, NOT per-page — chunk_parsed_doc detects the
        # mismatch with `pages` and skips page provenance (pages=[]) rather
        # than mis-attributing everything to page 1.
        page_markdown=[markdown],
        markdown=markdown,
        model="azure-document-intelligence",
        routes=["docintel"],
    )
