# Evaluation — does contextualization actually work?

Yes, and you can reproduce the measurement in one command. This page documents
the methodology, the results, and — just as important — the two things the
benchmark taught us that changed the package.

## TL;DR

On a corpus built to exhibit the ambiguous-chunk failure mode, measured with
`gpt-5.4-mini` + `text-embedding-3-large` (July 2026):

| Retrieval configuration | Pass@1 | Pass@3 | Pass@5 |
|---|---:|---:|---:|
| Plain chunks — dense only | 0.0% | 2.8% | 2.8% |
| Plain chunks — hybrid (dense + BM25 + RRF) | 0.0% | 0.0% | 2.8% |
| **Contextual** — dense only | 55.6% | 80.6% | 91.7% |
| **Contextual** — hybrid | 66.7% | 86.1% | 91.7% |
| **Contextual** — hybrid + LLM rerank | **91.7%** | **97.2%** | **97.2%** |

Retrieval failures at k=5 went from **97.2% (plain hybrid) to 2.8% (full
contextual stack)** — a 97% reduction. For scale reference, Anthropic's published evaluation
(737 real code chunks, 248 queries) reports Pass@10 rising 87.15% → 92.34%
with contextual embeddings, 93.21% adding BM25, and 95.26% adding a reranker
([cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)).
Our absolute numbers are more extreme because the corpus is *designed* to
maximize chunk ambiguity — theirs measures a natural mix.

## Methodology

`examples/benchmark.py` mirrors the shape of Anthropic's cookbook evaluation
(gold chunks, Pass@k) at demo scale:

- **Corpus** — 12 synthetic quarterly reports (3 fictional companies × 4
  quarters of 2025), ~10 chunks each, 120 chunks total. Documents are
  structurally identical; only the metrics differ — deterministically and
  *injectively* (no two documents ever share a metric value, so no two section
  bodies are ever byte-identical), and every run sees the same corpus. Crucially — and exactly like real filings — the
  section bodies **never repeat the company name or quarter**: "Total revenue
  grew 14% year over year…" appears in a chunk that carries no clue *whose*
  revenue it is. Only the title/overview names the company.
- **Queries** — 36 questions of the form *"How much did {company}'s revenue
  grow in {quarter} 2025?"* over three topics (revenue, operating margin,
  headcount) × all 12 documents.
- **Gold labels** — computed, not annotated: the gold set for a query is the
  chunks of the correct document whose section heading matches the topic.
- **Metric** — Pass@k: does any gold chunk appear in the top-k? Reported for
  k ∈ {1, 3, 5} over a top-5 ranking.
- **A/B isolation** — both stores contain the *identical* chunks; the only
  difference is whether the situating blurb was prepended for indexing. Same
  embedding model, same BM25, same fusion.

Why synthetic? Three reasons: gold labels are exact by construction (no
annotation noise), the corpus is public-safe and reproducible from a seedless
deterministic generator, and it isolates the *one* variable the technique
targets — chunk ambiguity. Treat the numbers as a demonstration of the failure
mode and its fix, not as a general-purpose retrieval leaderboard.

## Reading the table

**Plain retrieval doesn't degrade — it collapses.** With 12 near-identical
documents, the bare revenue chunks are interchangeable to the embedding model:
dense retrieval picks *a* revenue chunk, almost never the right one (0% P@1).
Hybrid doesn't rescue it: the query's company name matches only the
title/overview and boilerplate chunks — which contain the name but not the
metric — so BM25 promotes the wrong chunks and fusion inherits the confusion
(0% P@1, 2.8% P@5). This is the pathology Anthropic's blog describes,
reproduced cleanly.

**Blurbs restore the missing signal.** The contextualizer writes blurbs like
*"…from Borealis Industries' Q2 2025 quarterly report, in the Revenue
section…"*. Suddenly both retrievers can see company + quarter + topic:
dense-only jumps 2.8% → 91.7% P@5, and hybrid now *earns* its keep
(66.7% vs 55.6% P@1) because BM25 finally has the right exact terms to match.

**The reranker provides the decisive lift.** Fusion still confuses adjacent
quarters and sibling sections of the right document; one cheap listwise LLM
call over the top-30 fused candidates fixes most of it: P@1 66.7% → 91.7%,
P@3/P@5 → 97.2%.

## What the benchmark changed in the package

This evaluation wasn't decorative — two findings landed as code:

1. **Rerankers need the blurb too.** An early run showed the reranker
   *hurting* contextual hybrid (P@5 94% → 81% on that corpus revision): it was
   shown bare chunk text, which is ambiguous to a reranker for the same reason
   it was ambiguous to the retrievers — it reshuffled indistinguishable
   candidates. Reranking over `blurb + chunk` (as Anthropic's cookbook does)
   flipped it to the best condition. `rerank.py` now always ranks contextual
   text.
2. **Cache entries need a moment to settle.** An isolated probe (three
   sequential calls sharing a ~2.2k-token prefix) measured **1,792/2,190
   tokens (82%) served from cache** from the second call on — prefix caching
   works as designed. But on THIS corpus the aggregate hit rate is unstable
   (0–8% across runs, even ingesting sequentially): each document is only ~10
   quick calls, which can outrun the provider's few-second cache-settling
   window entirely. On real multi-page documents — dozens of chunks per
   document — the stream lasts long past settling and the steady-state rate
   dominates. `cached_tokens` in every ingestion report is there precisely so
   you can verify your own corpus.

   Also: a corpus-design bug found by review (colliding metric formulas made
   four Operating-margin sections byte-identical across documents) was fixed
   before the final measurement; the numbers above are from the
   collision-free corpus.

## The qualitative check

Same question to both pipelines — *"How much did Borealis Industries' revenue
grow in Q2 2025?"*:

> **Plain:** "The provided sources do not include Borealis Industries' revenue
> figures for Q2 2025, so I can't determine how much revenue grew from them
> alone. [S1] [S3]"
>
> **Contextual:** "Borealis Industries' total revenue grew **14% year over
> year** in Q2 2025, reaching **$595 million** for the quarter [S2]."

Note what the plain pipeline did: retrieval failed, and the grounding contract
made the model **refuse rather than hallucinate**. The correct answer (14%,
$595M) matches the corpus generator's ground truth exactly, with a citation
that resolves to the right document and section.

## Reproduce it

```bash
export OPENAI_API_KEY=sk-...
python examples/benchmark.py              # ~4 minutes, well under $1
python examples/benchmark.py --no-rerank  # skip the slowest condition
```

The script prints the Pass@k table, cache statistics, and the qualitative
side-by-side. It has no dependencies beyond the package itself.

## Limitations

- Synthetic, single-domain corpus built to exhibit one failure mode; absolute
  numbers will differ (be less extreme) on natural corpora.
- 36 queries — enough to separate 0% from 90%+, not to resolve ±3% deltas.
- Single run per condition (temperature 0 throughout keeps variance low).
- Pass@k measures retrieval only; the answer-quality check is qualitative.
