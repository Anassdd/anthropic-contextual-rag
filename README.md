# anthropic-contextual-rag

**[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) —
the technique published by Anthropic — as a complete, production-shaped RAG
package**: from raw PDFs to grounded, source-cited answers, against any
OpenAI-compatible endpoint.

> Independent implementation of the published technique, following Anthropic's
> [official cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).
> Not affiliated with or endorsed by Anthropic.

```
ingest:  PDF/Markdown ─▶ parse ─▶ chunk ─▶ contextualize ─▶ embed + BM25 ─▶ store
query:   question ─▶ [dense ‖ BM25] ─▶ fuse (RRF) ─▶ rerank ─▶ grounded answer [S#]
```

## Why

A chunk pulled out of its document loses its context — *"Revenue grew 12%"* —
whose revenue? which quarter? Embeddings and keyword search index the chunk as
written, so they can't tell. Contextual Retrieval fixes this: before indexing,
an LLM writes a short blurb situating each chunk within its parent document.
The blurb is prepended **for indexing only**; the original chunk text is what
gets cited. Prompt-prefix caching makes it affordable — the document rides at
the start of every blurb prompt and is billed at the cached rate after the
first chunk (**82% of prompt tokens cached**, measured here).

**Measured** ([full methodology & analysis](docs/evaluation.md)) — same chunks,
same models, same fusion; the only difference is the blurb:

| Retrieval configuration | Pass@1 | Pass@5 |
|---|---:|---:|
| Plain chunks — hybrid (dense + BM25 + RRF) | 0.0% | 2.8% |
| **Contextual** — hybrid | 66.7% | 91.7% |
| **Contextual** — hybrid + LLM rerank | **91.7%** | **97.2%** |

On Anthropic's natural-corpus evaluation the technique cuts retrieval failures
**49%** (67% with reranking); our corpus is built to maximize the
ambiguous-chunk failure mode, so the gap is starker. Reproduce it:
`python examples/benchmark.py`.

## Quickstart

```bash
pip install "anthropic-contextual-rag[pdf] @ git+https://<your-git-host>/<you>/anthropic-contextual-rag.git"
export OPENAI_API_KEY=sk-...
```

```python
from contextual_rag import ContextualRAG

rag = ContextualRAG("my-corpus")                 # named, persistent collection

rag.ingest_pdf("report.pdf")                     # parse → chunk → contextualize → index
rag.ingest_markdown(text, doc_id="notes.md")     # or skip parsing entirely

hits = rag.search("What drove Q3 margins?")      # hybrid + reranked, best first
ans = rag.ask("What drove Q3 margins?")          # grounded, [S#]-cited answer
print(ans.text)                                  # "... margins rose 3pts [S1] ..."
print(ans.sources[0].citation)                   # "report.pdf, p.12"
```

Point it at a corporate gateway instead of OpenAI with one variable —
`OPENAI_BASE_URL=https://gateway.example.com/v1`. Every knob is documented in
the [usage guide](docs/usage.md).

## What's in the box

- **Structure-aware chunker** — cuts on headings → paragraphs → sentences,
  ~512-token targets, sentence-snapped overlap; code/math/tables are atomic
  (never split); every chunk carries doc + page + section provenance.
- **Contextualizer** — Anthropic's prompt verbatim, document-first for prefix
  caching, cache-priming concurrency, and **excerpt mode** for documents too
  large for the model's input window (a case the published recipe doesn't
  handle) — head + region excerpts, batched so caching keeps working.
- **Hybrid retrieval** — Chroma (embedded, no server) + pure-Python BM25 over
  the same contextual text, fused with Reciprocal Rank Fusion; per-collection
  index caching so queries never pay a rebuild.
- **Reranking** — `llm` (one RankGPT-style call, default), `endpoint` (hosted
  cross-encoder), or `off`. Rerankers see *blurb + chunk* — measurably better
  than bare chunks ([why](docs/evaluation.md)).
- **Grounded answers** — constrained to retrieved sources, inline `[S#]`
  citations resolvable to document + page, refuses rather than guesses,
  answers in the question's language.
- **PDF parsing (optional)** — per-page tiered routing: free text layer for
  clean prose, vision LLM (Markdown + LaTeX, figures *extracted* not labeled)
  for everything else; Azure Document Intelligence as an alternative backend.
- **One endpoint seam** — a single module speaks the OpenAI API for the whole
  package; `search_trace()` exposes every retrieval stage for debugging;
  offline test suite (no keys needed); fully typed (`py.typed`).

## Documentation

| | |
|---|---|
| [Usage guide](docs/usage.md) | Install, configuration, the 5-minute tour, recipes, cost model |
| [How it works](docs/how-it-works.md) | The pipeline stage by stage, design decisions, extension seams |
| [Evaluation](docs/evaluation.md) | Methodology, measured results, what the benchmark changed in the code |
| [Changelog](CHANGELOG.md) | Versions and notable changes |

## Project layout

```
src/contextual_rag/
├── rag.py           # ContextualRAG facade — the one-object API
├── config.py        # env/.env/configure() → Settings (lazy: import needs no key)
├── llm.py           # THE endpoint seam — only file importing the OpenAI SDK
├── chunker.py       # structure-aware Markdown chunker
├── contextual.py    # the contextualizer (whole-doc + excerpt modes)
├── store.py         # Chroma vector store (embedded, persistent, per-collection)
├── bm25.py          # pure-Python Okapi BM25
├── search.py        # dense ‖ BM25 → RRF → rerank, with per-stage trace
├── rerank.py        # llm / endpoint / off
├── answer.py        # grounded, cited generation
├── ingest.py        # parse → chunk → contextualize → store, one call
├── index_cache.py   # per-collection records+BM25 cache
├── types.py         # Chunk, ScoredChunk, RetrievalTrace, Answer
└── parsing/         # optional: tiered vision parser + Azure Document Intelligence

examples/quickstart.py    # smallest end-to-end run
examples/benchmark.py     # the contextual-vs-plain A/B measurement
tests/                    # offline suite — LLM and embeddings faked, no keys
```

## Tests

```bash
pip install -e ".[dev]"
pytest        # offline — LLM and embeddings faked, a few seconds
ruff check .  # lint-clean
```

## Credits & license

The contextualization prompt and technique follow Anthropic's
[Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval)
and [official cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).
The excerpt mode, tiered PDF parsing, cache-aware concurrency, defensive
reranking, and the retrieval trace are this package's own. Not affiliated with
Anthropic.

License: not yet chosen — until a LICENSE file lands, all rights reserved.
