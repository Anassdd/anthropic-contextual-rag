# Prompt caching — the whole trick

Contextual retrieval sends the *entire parent document* with every single blurb
call. Read that naïvely and the technique looks unaffordable; read it with
prompt caching in mind and it costs a tenth as much. This page exists because
caching is where people lose money without noticing, and because "is it
actually working?" has a concrete answer you can check.

If you only remember three things:

1. Caching applies to a **byte-identical prompt prefix** — same opening tokens,
   exactly, from token zero.
2. This library's prompts are engineered so the document *is* that prefix.
3. Every ingestion report carries `cached_tokens`. **Look at it.** On a
   multi-page document, expect a large fraction of `context_tokens`; near-zero
   means you're paying full price and something on this page explains why.

## What prompt caching actually is

Every time you call an LLM, the model processes your entire prompt from the
first token before it generates anything. Imagine a colleague who, for every
question you ask about a 300-page manual, insists on re-reading the manual from
page one. Prompt caching is the provider noticing: *"the first 15,000 tokens of
this request are identical to one I processed 20 seconds ago — I still have
that computation"* — and skipping the re-read. You get faster responses, and
the provider passes part of the saving back as a discount on those cached
tokens.

The rules that follow from "it's a prefix cache":

- The match starts at **token zero**. A single changed byte early in the
  prompt (a timestamp, a document ID, a different system message) means
  nothing after it can hit the cache.
- There's a **minimum prefix length** before caching engages at all —
  typically 1,024 tokens. A one-page document never gets cached; that's fine,
  it's also cheap to resend.
- Cache entries are **short-lived** — minutes of inactivity, not hours. This
  matters for how you schedule your calls, not just how you write your prompt.

## Why contextual retrieval is the perfect caching customer

For one document with 30 chunks, this library makes 30 calls that all begin
with the same thing:

```
<document>
...the entire document, byte-identical every time...
</document>

Here is the chunk we want to situate...   ← only this part changes
```

The document rides *first* in the prompt — that ordering is Anthropic's
published design and it is not cosmetic. It means the expensive part of every
call (the document) is the cacheable part, and the changing part (the chunk) is
the cheap tail. The document is billed in full exactly once; the other 29 calls
pay the cached rate for it.

Worked numbers (full derivation in [costs.md](costs.md)): a 100-document,
3,000-chunk corpus costs **~$12 to contextualize uncached** and **~$2.30
cached** at mini-tier prices. Same calls, same output — the difference is
purely whether the prefix hits.

When a document is too large to send whole, excerpt mode batches consecutive
chunks so each batch shares one byte-identical excerpt — caching keeps working
per batch instead of per document. You don't configure any of this; it's how
the prompts are built.

## Provider guide

The library speaks the OpenAI API, so OpenAI and Azure OpenAI are the direct
paths. Anthropic's caching model is included both because the technique is
theirs and because many corporate gateways translate to Claude behind an
OpenAI-shaped façade.

### OpenAI

- **Automatic.** No flag, no code change, no write surcharge. Prefixes of
  **1,024+ tokens** are cached (hits extend in 128-token steps).
- **Discount depends on the model family** — roughly 90% off cached input on
  current (gpt-5-class) models, 75% on the gpt-4.1/o-series generation, 50% on
  gpt-4o-generation. Check the pricing page for your exact model; the library
  benefits automatically either way.
- **Lifetime:** entries typically survive 5–10 minutes of inactivity (up to
  ~an hour off-peak). Fresh use refreshes them.
- **Verify:** every response reports
  `usage.prompt_tokens_details.cached_tokens` — the library surfaces this in
  its ingestion reports.
- At very high volume, OpenAI's `prompt_cache_key` parameter helps route
  identical prefixes to the same cache shard; at this library's scale you
  won't need it.

Reference: [OpenAI prompt caching guide](https://platform.openai.com/docs/guides/prompt-caching).

### Azure OpenAI

Same mechanism as OpenAI — automatic, 1,024-token minimum, `cached_tokens` in
the usage block — but deployment realities add three surprises worth knowing
*before* you deploy:

- **Only newer model deployments cache.** `gpt-4o` (2024-08-06+), `gpt-4o-mini`,
  the o-series, and later families support it. A corporate gateway pinned to a
  legacy `gpt-4` or `gpt-35-turbo` deployment gets **zero caching** — your
  ingestion silently costs the full uncached price. If `cached_tokens` is 0
  everywhere, check the deployed model version first.
- **The discount depends on your deployment type.** Standard deployments get a
  discounted cached-input rate (mirroring OpenAI's percentages); provisioned
  (PTU) deployments have gone further — cached tokens at up to a 100% discount
  and not counting against throughput. Confirm on current Azure pricing for
  your region and type.
- **Load balancers can defeat the cache.** The cache lives per deployment. An
  API-Management/gateway layer that round-robins requests across regional
  deployments sends call 2 of your document to a backend that never saw call 1.
  If you front Azure with a balancer, route ingestion traffic sticky (one
  document's calls to one backend), or accept uncached pricing.
- **Report visibility:** `cached_tokens` appears from API version
  `2024-10-01-preview` onward; an older pinned `api-version` hides the field
  (the caching still works — you just can't see it).

Reference: [Azure OpenAI prompt caching](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching).

### Anthropic (Claude)

Anthropic's caching is **explicit** rather than automatic, and it's the model
the original cookbook uses:

- You mark the end of the cacheable prefix with a `cache_control:
  {"type": "ephemeral"}` breakpoint on a content block — in the contextual
  retrieval recipe, on the document block, so each per-chunk call reuses it.
- **Pricing is asymmetric:** writing a cache entry costs 1.25× the base input
  rate; *reading* it costs 0.10× — a 90% discount, refreshed on each use.
  The default entry lives 5 minutes from last use; a 1-hour TTL is available
  at a higher write multiplier.
- **Minimums:** 1,024 tokens (2,048 on Haiku-class models). Up to 4
  breakpoints per request.
- Because this library speaks the OpenAI protocol, running it against Claude
  means going through a translating gateway (LiteLLM and similar map
  OpenAI-shaped requests onto Anthropic caching) — or implementing the
  technique natively from Anthropic's own materials, which are excellent:
  the [contextual retrieval post](https://www.anthropic.com/news/contextual-retrieval),
  the [cookbook notebook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide),
  and the [prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

### At a glance

| | OpenAI | Azure OpenAI | Anthropic |
|---|---|---|---|
| Activation | automatic | automatic (supported deployments only) | explicit `cache_control` |
| Cached read price | ~10–50% of input, by family | discounted; up to free on provisioned | 10% of input |
| Cache write surcharge | none | none | 1.25× (2× for 1-hour TTL) |
| Minimum prefix | 1,024 tokens | 1,024 tokens | 1,024 (2,048 Haiku) |
| Typical lifetime | 5–10 min idle | 5–10 min idle | 5 min, refreshed (1 h option) |
| Verify via | `prompt_tokens_details.cached_tokens` | same (recent `api-version`) | `cache_read_input_tokens` |

## How this library keeps the cache hot

Things already done for you, worth knowing so you don't undo them:

- **Document-first prompts.** The cacheable content leads; nothing volatile
  (no timestamps, no chunk ids) appears before it.
- **Prime, then fan out.** The first chunk of each document is contextualized
  *alone* — that call writes the cache entry — and only then does a small
  thread pool run the rest in parallel. Fanning out immediately would make
  every worker pay the full uncached prefix simultaneously.
- **Batch-aligned excerpts.** Oversized documents share byte-identical
  excerpts per batch of consecutive chunks, so the same economics apply.

The one knob that's yours: **`CONTEXT_CONCURRENCY`**. After the priming call,
this many workers run at once (default 4). Providers need a few seconds for a
fresh cache entry to become readable — on a *short* document (a handful of
quick calls), a parallel burst can finish before the entry settles, and you'll
see low `cached_tokens` despite doing everything right. Set it to `1` for
short-document corpora; keep the default for multi-page documents, where the
call stream lasts well past settling. (Measured here: 82% of prompt tokens
cached in steady state; the same code on 10-chunk mini-docs wobbles between
0–8% purely from settling. The [evaluation](evaluation.md) tells that story.)

## Is it working? A checklist

Ingest one real multi-page document and read the report:

```python
report = rag.ingest_markdown(text, doc_id="report.md")
print(report["context_tokens"], report["cached_tokens"])
# healthy: cached_tokens is a large fraction of context_tokens
```

If `cached_tokens` is (near) zero:

1. **Document under ~1,024 tokens?** Below the minimum prefix — expected, and
   cheap anyway.
2. **Short document + parallel workers?** The settling race. Try
   `CONTEXT_CONCURRENCY=1`.
3. **On Azure?** Check the deployed model version supports caching, and that
   your `api-version` is new enough to *report* it.
4. **Behind a gateway/load balancer?** If it rewrites prompts (injects
   headers, system prefixes that vary per call) or round-robins backends, the
   prefix never matches. Ask your gateway team what's between you and the
   model.
5. **Long pauses between calls?** Entries expire after minutes of inactivity.
   Ingest a document's chunks in one go (the library already does) rather
   than trickling them.

One non-obvious consequence of "minutes of lifetime": re-running ingestion an
hour later re-pays the full prefix once per document. That's by design —
caching makes each document's *burst* of calls cheap; it is not long-term
storage.
