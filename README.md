# anthropic-contextual-rag

**[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (the
technique published by Anthropic) as a complete, reusable RAG package** — from raw
documents to grounded, source-cited answers, against any OpenAI-compatible endpoint.

> This is an independent implementation of the published technique. It is not
> affiliated with or endorsed by Anthropic.

```
ingest:  PDF/Markdown ─▶ parse ─▶ chunk ─▶ contextualize ─▶ embed + BM25 ─▶ store (Chroma)
query:   question ─▶ [dense ‖ BM25] ─▶ fuse (RRF) ─▶ rerank ─▶ top-k ─▶ grounded answer [S#]
```

## Why contextual retrieval

A chunk pulled out of its document loses context ("the model" — which model? "this
approach" — which?). Embeddings and BM25 then match it poorly. Contextual Retrieval
fixes that: before indexing, an LLM writes a short blurb situating each chunk within
its parent document, and that blurb is prepended to the chunk for embedding and
keyword indexing — while the **original** chunk text is what gets cited. Anthropic's
first-party result: **−49%** retrieval failures (contextual embeddings + contextual
BM25), **−67%** with a reranker.

The trick that makes it affordable: the whole document is sent on every chunk call but
sits at the **start** of the prompt, so repeated calls hit automatic prompt-prefix
caching — the document is billed in full once, then at the provider's cached rate.

## What's in the box

- **Structure-aware Markdown chunker** — cuts on headings, then paragraphs/sentences,
  ~512-token targets with sentence-snapped overlap. Code fences, display math and HTML
  tables are atomic (never split). Every chunk carries provenance: doc, pages, section.
- **Contextualizer** — Anthropic's prompt verbatim, document-first for prefix caching,
  cache-aware concurrency (the first call per prefix group runs alone to prime the
  cache). **Excerpt mode** for documents too large for the model's input window (see
  below) — a case the original recipe doesn't handle.
- **Hybrid retrieval** — dense (Chroma, embedded/on-disk, no server) + pure-Python
  Okapi BM25 over the same contextualized text, fused with Reciprocal Rank Fusion.
  The query-side record list and BM25 index are cached per collection and invalidated
  on writes — never rebuilt per query.
- **Reranker seam** — `llm` (RankGPT-style, one cheap chat call — the default),
  `endpoint` (hosted cross-encoder, Cohere/Jina shape), or `off`.
- **Grounded answers** — constrained to retrieved sources with inline `[S1]`-style
  citations resolvable to doc + page; answers in the question's language.
- **PDF parsing (optional)** — per-page tiered routing: clean prose pages use the free
  text layer; pages with math, tables, figures, or a garbled text layer are rendered
  and transcribed by a vision LLM to Markdown + LaTeX (figures are *extracted*, not
  just labeled). Azure Document Intelligence is available as an alternative backend.
- **One endpoint seam** — every LLM/embedding call goes through a single module
  speaking the OpenAI API. Standard OpenAI, an Azure-hosted proxy, LiteLLM, vLLM, or a
  corporate gateway are all just `OPENAI_BASE_URL` — a config change, never a code change.

## Install

```bash
pip install "anthropic-contextual-rag @ git+https://<your-git-host>/<you>/anthropic-contextual-rag.git"

# with PDF ingestion (vision parser):
pip install "anthropic-contextual-rag[pdf] @ git+..."

# everything:
pip install "anthropic-contextual-rag[pdf,docintel,tokens] @ git+..."
```

Or clone and `pip install -e ".[pdf,tokens,dev]"`.

## Quickstart

```python
from contextual_rag import ContextualRAG

rag = ContextualRAG("my-corpus")                 # a named, persistent collection

rag.ingest_pdf("report.pdf")                     # parse → chunk → contextualize → index
rag.ingest_markdown(text, doc_id="notes.md")     # or skip parsing entirely

hits = rag.search("What drove Q3 margins?", k=8)      # ranked ScoredChunks
trace = rag.search_trace("What drove Q3 margins?")    # every stage: dense/bm25/fused/reranked

ans = rag.ask("What drove Q3 margins?")               # grounded answer
print(ans.text)                                       # "... rose 3pts [S1] ..."
for s in ans.sources:
    print(s.citation)                                 # "report.pdf, p.12"
```

Configuration comes from env vars / a `.env` file (see `.env.example`), or from code:

```python
from contextual_rag import configure
configure(api_key="sk-...", chat_model="gpt-5.4-mini", embed_model="text-embedding-3-large")
```

Every step is also a plain function if you'd rather own the store:

```python
from contextual_rag import chunk_markdown, contextualize_chunks, VectorStore, search, answer_from

chunks = chunk_markdown(doc_text, doc_id="doc.md")
ctx = contextualize_chunks(doc_text, chunks)     # one LLM call per chunk, prefix-cached
store = VectorStore("corpus"); store.add(ctx)
final = search("my question", store=store, rerank_mode="llm")
print(answer_from("my question", final).text)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (may be blank for keyless gateways) |
| `OPENAI_BASE_URL` | *(standard OpenAI)* | Any OpenAI-compatible endpoint |
| `CHAT_MODEL` | `gpt-5.4-mini` | Blurbs, LLM rerank, answers |
| `EMBED_MODEL` | `text-embedding-3-large` | Embeddings (multilingual-strong) |
| `PARSE_MODEL` | *(falls back to chat)* | Vision model for PDF pages |
| `CONTEXT_DOC_CAP` | `250000` | Whole-document mode limit (tokens) — see below |
| `CONTEXT_PART_TOKENS` | `48000` | Excerpt size in excerpt mode |
| `CONTEXT_CONCURRENCY` | `4` | Parallel blurb calls per prefix group |
| `RETRIEVAL_RERANK` | `llm` | `llm` \| `endpoint` \| `off` |
| `RERANK_MODEL/BASE_URL/API_KEY` | — | Hosted cross-encoder (`endpoint` mode) |
| `VECTOR_DIR` | `./.contextual_rag` | Where the vector store persists |
| `PARSER` | `vision` | `vision` \| `docintel` |
| `DOCINTEL_ENDPOINT/KEY` | — | Azure Document Intelligence credentials |

**When changing the chat model, resize `CONTEXT_DOC_CAP`**: keep it ≈20k tokens under
the model's usable input (250k fits the GPT-5 family; a 128k-context model needs
~100k). Too high = hard API errors mid-ingestion; too low = merely more excerpts.

## Oversized documents — excerpt mode

Anthropic's recipe silently assumes the document fits the model's input; a 500k-token
SEC filing doesn't (and blurb quality degrades with irrelevant bulk — "context rot").
Documents over `CONTEXT_DOC_CAP` are handled automatically: each chunk is situated
against the document **head** (~6k tokens — title, TOC, intro: the document's identity)
plus the **region around the chunk**, totalling ~`CONTEXT_PART_TOKENS`. Consecutive
chunks are batched — aligned to top-level section boundaries — to share one
byte-identical excerpt, so prefix caching keeps working per batch exactly as it does
per document. Documents under the cap get Anthropic's recipe untouched.

## Cost notes

- Contextualization is **one LLM call per chunk**, at ingestion time only. The
  document prefix is cached after the first chunk, so the marginal cost per chunk is
  roughly the blurb tokens plus the cached-rate prefix. `cached_tokens` is surfaced in
  every ingestion report so you can verify caching is working.
- A cheap-tier chat model is fine for blurbs; the pipeline defaults everything to
  `CHAT_MODEL` and lets you override per call (`context_model=`).
- `llm` reranking adds exactly one chat call per query (skipped automatically when the
  candidate pool already fits in `k`).

## Project layout

```
src/contextual_rag/
├── rag.py           # ContextualRAG facade
├── config.py        # env/.env → Settings (lazy; configure() for code overrides)
├── llm.py           # the ONLY module importing the OpenAI SDK: chat/embed/vision
├── types.py         # Chunk, ScoredChunk, RetrievalTrace, Answer
├── chunker.py       # structure-aware Markdown chunker
├── tokens.py        # tiktoken if available, heuristic otherwise
├── contextual.py    # the contextualizer (whole-doc + excerpt modes)
├── store.py         # Chroma vector store (embedded, persistent)
├── bm25.py          # pure-Python Okapi BM25
├── index_cache.py   # per-collection records+BM25 cache
├── search.py        # dense ‖ BM25 → RRF → rerank (with per-stage trace)
├── rerank.py        # llm / endpoint / off
├── answer.py        # grounded, cited generation
├── ingest.py        # parse → chunk → contextualize → store, one call
└── parsing/         # optional: vision parser + Azure Document Intelligence
```

## Tests

Offline — no network, no keys, LLM and embeddings faked:

```bash
pip install -e ".[dev]"
pytest
```

## Credits & license

The contextualization prompt and technique follow Anthropic's
[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) post; the
excerpt mode, tiered PDF parsing, cache-aware concurrency, and the retrieval stack
around it are this package's own. Not affiliated with Anthropic.

License: not yet chosen — until a LICENSE file lands, all rights reserved.
