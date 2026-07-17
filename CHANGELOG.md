# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

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
