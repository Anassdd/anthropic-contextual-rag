# What it costs — and when plain RAG is the better call

If you're evaluating this library, the first honest question is: *"why not just
do normal RAG?"* Fair. This page answers it with real numbers, and it ends with
the cases where the answer is "you shouldn't use this library." Read it before
you commit.

**The one-sentence version:** contextual retrieval costs extra exactly once, at
ingestion (one LLM call per chunk, made affordable by prompt caching), costs
*nothing* extra per query, and buys you a large retrieval-quality jump on
corpora whose chunks are ambiguous out of context — which is most long-document
corpora, and not most FAQ-style ones.

## What plain RAG costs

A plain RAG pipeline pays for two things at ingestion: nothing (chunking is
local string surgery) and embeddings. Embeddings are cheap. A 3,000-page corpus
is roughly 1.5 million tokens; at typical embedding prices (~$0.13 per million
tokens for `text-embedding-3-large`) that's about **$0.20**. Twenty cents for
the whole corpus.

At query time it pays for an embedding call (fractions of a cent), and whatever
your answer generation costs. That part is identical in both pipelines, so we
can ignore it for the comparison.

## What contextual RAG adds

One chat call **per chunk**, at ingestion. Each call sends the *entire parent
document* plus the chunk, and gets back a one-sentence situating blurb. That
"entire parent document, every time" is the scary part, so let's do the math on
a realistic corpus instead of hand-waving.

### The worked example

Take a corpus you might actually have:

| | |
|---|---:|
| Documents | 100 reports |
| Length | ~30 pages each (~500 tokens/page) |
| Tokens per document | ~15,000 |
| Corpus total | ~1.5M tokens |
| Chunks (at ~512-token targets) | ~30 per doc, **3,000 total** |

Each blurb call sends: the document (15,000 tokens) + the chunk (~512) + the
instructions (~90) ≈ **15,600 input tokens**, and returns a ~60-token blurb.
Example prices below are a mini-tier chat model at **$0.25 per million input
tokens** ($2 per million output) — check your provider's current sheet, the
*ratios* are what matter.

**Without caching** (the naïve reading):

    3,000 calls × 15,600 tokens ≈ 47M input tokens  →  ~$11.70
    3,000 × 60 output tokens    ≈ 180k tokens       →  ~$0.36
                                                       ─────────
                                                       ~$12

**With prompt caching** (what actually happens — see [caching.md](caching.md)):
the document sits at the *start* of every prompt, byte-identical across all 30
of a document's calls. The provider caches that prefix after the first call and
bills it at ~10% for the other 29:

    Per document:
      call 1:       15,600 tokens at full price
      calls 2–30:   ~600 tokens full price (chunk + instructions)
                  + ~15,000 tokens at the ~10% cached rate
    Whole corpus:   ≈ 7.7M billed-token-equivalents  →  ~$1.90
    + output                                         →  ~$0.36
                                                        ─────────
                                                        ~$2.30

So for this corpus:

| | Ingestion cost | Per-query cost |
|---|---:|---:|
| Plain RAG | ~$0.20 | baseline |
| Contextual, no caching | ~$12 | baseline |
| **Contextual, with caching** | **~$2.30** | **baseline (identical)** |

Caching turns a ~60× premium over plain RAG into a ~12× premium — and the
absolute number for three thousand pages is **a couple of dollars, once**. Rule
of thumb at mini-tier prices: **under $1 per 1,000 pages** ingested, cached.

Two things scale this number: document *length* (the prefix you resend) and the
cached discount your provider gives (90% on current OpenAI models, less on
older families — details in [caching.md](caching.md)). If caching is
unavailable to you, the uncached column is your real cost; for long documents
that can genuinely change the decision, which is why the caching page exists.

### Query time: nothing changes

This surprises people, so plainly: **a contextual query costs the same as a
plain one, and takes the same time.** Same single embedding call, same local
BM25 scan, same fusion. The blurbs live in the index; the answer model is shown
the *original* chunks either way, so the answer prompt is byte-for-byte the
same size. (The optional LLM reranker sees blurb+chunk snippets, adding a few
hundred tokens to that one call — noise.) The entire premium is one-time, at
ingestion. Re-ingestion of an *unchanged* document is skippable
(`rag.doc_ids()`), so you don't pay it twice by accident.

## What you get for it

Anthropic's published evaluation (a natural corpus: 737 code chunks, 248
queries) measured retrieval failures at top-20 dropping **35%** with contextual
embeddings alone, **49%** adding contextual BM25, and **67%** adding a reranker
([source](https://www.anthropic.com/news/contextual-retrieval)).

Our own [benchmark](evaluation.md), on a synthetic corpus *built* to maximize
chunk ambiguity (12 near-identical quarterly reports), is starker: plain hybrid
retrieval finds a correct chunk in its top-5 just 2.8% of the time; the full
contextual stack, 97.2%. Real corpora land between those two results —
closer to Anthropic's numbers the more self-contained your chunks already are.

Be clear-eyed about *where* the gain lives. It is large when documents state
their identity once (title page, opening clause) and then spend fifty pages
saying "the Company", "the quarter", "the patient" — filings, contracts,
reports, manuals, papers — and your users ask questions that *name* those
entities. It is marginal when each chunk already says what it's about.

**A 60-second test on your own corpus:** pull ten random chunks and read them
cold. For each, ask "could a stranger tell what document this is from and what
it's about?" Mostly no → this technique will move your numbers a lot. Mostly
yes → it will move them a little, and you should weigh that against the
ingestion cost above.

## When plain RAG (or no RAG) is genuinely the better call

- **Your chunks are already self-contained.** FAQ entries, product cards,
  short standalone notes, tickets. The blurb would restate what the chunk
  already says. Save your money.
- **Your whole corpus fits in one prompt.** Under ~100k tokens of documents?
  Skip retrieval entirely and put the corpus in context. RAG of any kind is
  machinery you don't need yet.
- **Your corpus churns constantly.** The ingestion premium is "one-time" only
  if documents are ingested roughly once. Re-contextualizing an
  hourly-refreshed feed multiplies the cost by every refresh.
- **You can't get prompt caching.** On an endpoint without it (some older
  Azure deployments, some gateways — see [caching.md](caching.md)) the
  uncached column is your price. For short documents that's still fine; for
  500-page documents it isn't.
- **You need the simplest thing that works, today.** Plain hybrid RAG is this
  same library with the contextualizer skipped (`context=""`), and it's one
  less LLM dependency at ingestion. Start there, run the ten-chunk test, and
  upgrade if retrieval actually misses.

You can hold both pipelines in your hand here: `examples/benchmark.py` builds
the same corpus with and without blurbs and prints the head-to-head numbers,
and `examples/offline_demo.py` shows the difference with no API key at all.
Deciding *not* to use the technique, on evidence, is a fine outcome.
