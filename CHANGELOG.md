# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-07-20

An engineering-review release: a fresh end-to-end audit of the whole library,
fixing the correctness issues it found and hardening the failure modes around
configuration. No retrieval behavior changed for well-formed inputs — the
published benchmark numbers in `docs/evaluation.md` remain valid.

### Added
- `examples/offline_demo.py` — the full pipeline end to end with **no API key
  and no network**: the endpoint seam is faked in ~60 lines, everything else
  runs for real, and the script ends in hard assertions (it doubles as an
  end-to-end smoke test of a checkout). Includes a miniature plain-vs-contextual
  A/B that reproduces the ambiguous-chunk failure mode.
- `tests/test_usage.py` — the documented usage patterns (README quickstart,
  doc-scoped questions, resumable ingestion, collection isolation, the
  functional no-facade pipeline) as offline tests: if one fails, the docs lie.
- `tests/test_config.py` — configuration loading, precedence, and validation.
- `ContextualRAG.ask(..., doc_id=...)` — scope a question to one document
  (search already supported it; ask now passes it through).

### Changed
- **Markdown pipe tables are now atomic** in the chunker, like HTML tables:
  the vision parser emits tables in pipe form, and an oversized one used to be
  sentence-split — tearing rows mid-table. Regression-tested.
- `configure()`/env values for `retrieval_rerank` and `parser` are validated:
  a typo like `RETRIEVAL_RERANK=none` used to silently select the **LLM**
  reranker (surprise cost on every query); it now raises `ConfigError` at
  load. `rerank()` likewise rejects unknown modes.
- Importing the package no longer has side effects: `.env` is read on first
  settings load, not at import time.
- `ContextualRAG.ingest_pdf(bytes)` now requires `filename`: the old default
  (`"document.pdf"`) made two byte-source ingestions silently overwrite each
  other, since chunk ids derive from the doc id.

### Fixed
- Endpoint (cross-encoder) rerank scores survive: `_apply_order` used to
  overwrite the remote `relevance_score` with reciprocal rank, so
  `scores["rerank"]` lied about what the reranker said.
- Malformed numeric env vars (`CONTEXT_DOC_CAP=abc`, or set-but-empty) raise a
  `ConfigError` naming the variable, instead of a bare `ValueError` (or crash)
  from deep inside settings loading.
- Chunks now carry the `domain_id` of the collection they are stored in;
  `ingest_*` used to stamp `"default"` regardless of the target collection.
- The store's cache identity uses the *resolved* path, so two spellings of one
  directory (`./x`, `x`, `~/x`) share one index-cache entry instead of two.

## [0.2.0] — 2026-07-17

### Added
- Full API documentation: Google-style docstrings on every public function,
  `docs/usage.md` (user guide) and `docs/how-it-works.md` (architecture).
- `docs/evaluation.md` — a reproducible contextual-vs-plain benchmark
  (`examples/benchmark.py`) with measured results.
- `py.typed` marker — the package now ships its type hints (PEP 561).
- Ruff lint configuration; the codebase is lint-clean under
  `E, W, F, I, B, UP, SIM`.
- Project URLs (Anthropic's technique blog post and official cookbook).

### Changed
- Docstrings and comments rewritten for clarity throughout (the 0.1.0 test
  suite passes unchanged).
- Rerankers (both `llm` and `endpoint` modes) now rank over the contextual
  text (blurb + chunk) instead of the bare chunk — measured on the benchmark,
  reranking bare chunks *hurt* (P@5 94%→81%) while reranking contextual text
  reaches P@3/P@5 100%. Matches the approach in Anthropic's cookbook.

### Fixed
- `ContextualRAG.ask()` raised `AttributeError` — the facade resolved
  `answer`/`search` through package attributes that the package `__init__`
  shadows with same-name function re-exports. Latent since 0.1.0 (no test
  covered `ask()`); now fixed with direct submodule imports and pinned by a
  regression test.
- Azure Document Intelligence parses no longer stamp **page 1** on every chunk
  of a multi-page document: a parser returning one unified Markdown blob now
  yields `pages=[]` (citations honestly omit the page) instead of wrong
  provenance. Found by adversarial review; regression-tested.
- Collection names: unusual `domain_id`s (trailing punctuation, `..`,
  non-ASCII) produced Chroma-invalid names that crashed `VectorStore`
  construction, and sanitization could silently merge distinct ids
  (`"a b"`/`"a_b"`). Names are now validated and collision-proofed with a
  hash suffix; clean ids keep their historical collection names.
- Hosted-reranker (`endpoint` mode) responses with out-of-range or negative
  indices crashed the query or mis-stamped scores; invalid indices are now
  dropped defensively.
- `index_cache`: a read racing an in-process write can no longer re-cache a
  stale snapshot (generation counter); the cross-process freshness guarantee
  is now stated honestly (count changes are detected, same-count upserts from
  *other processes* are not).
- Excerpt mode now honors small `CONTEXT_PART_TOKENS` budgets by scaling the
  head/margin/span anatomy down proportionally (previously excerpts floored at
  ~22k tokens regardless of the setting).
- Dependency floors raised to versions the code actually requires:
  `openai>=1.50` (the `max_completion_tokens` retry) and `pypdfium2>=5`
  (`PdfObject.get_bounds`). Both verified by installing the old floors.

## [0.1.0] — 2026-07-17

### Added
- Initial release: structure-aware Markdown chunker, Anthropic-style
  contextualizer (whole-document + excerpt modes, prefix-cache-aware
  concurrency), Chroma vector store, pure-Python BM25, RRF fusion,
  llm/endpoint/off reranker seam, grounded `[S#]`-cited answers.
- Optional PDF ingestion: tiered vision parser (`[pdf]` extra) and Azure
  Document Intelligence (`[docintel]` extra).
- `ContextualRAG` facade; offline test suite (32 tests, no keys required).
