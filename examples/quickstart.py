"""Quickstart — ingest a Markdown document and ask a grounded question.

Needs an endpoint configured (see .env.example):
    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

from contextual_rag import ContextualRAG

DOC = """# Contextual Retrieval

## The problem
A chunk pulled out of its document loses context: "the model" — which model?
"this approach" — which? Embeddings and keyword search then match it poorly.

## The fix
Before indexing, an LLM writes a one- or two-sentence blurb situating each chunk
within its parent document. The blurb is prepended to the chunk for embedding and
BM25 — but the original chunk text is what gets cited.

## Results
Anthropic reported contextual embeddings + contextual BM25 cut retrieval failures
by 49 percent, and 67 percent when combined with a reranker.
"""

rag = ContextualRAG("quickstart")

if rag.count() == 0:  # ingestion is idempotent per doc, but skip the LLM cost on re-runs
    report = rag.ingest_markdown(DOC, doc_id="contextual-retrieval.md")
    print(f"ingested: {report['chunks']} chunks, "
          f"{report['context_tokens']} contextualizer tokens\n")

# a PDF works the same way (needs the [pdf] extra):
#   rag.ingest_pdf("report.pdf")

answer = rag.ask("How much does contextual retrieval improve retrieval?")
print(answer.text)
print("\nSources:")
for i, s in enumerate(answer.sources, 1):
    print(f"  [S{i}] {s.citation} — {s.section}")
