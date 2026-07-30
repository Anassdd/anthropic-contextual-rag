# Usage guide

Everything you need to go from `pip install` to grounded, cited answers.
For *why* the pipeline is built this way, read [how-it-works.md](how-it-works.md);
for measured results, read [evaluation.md](evaluation.md).

## Install

```bash
pip install "anthropic-contextual-rag @ git+https://<your-git-host>/<you>/anthropic-contextual-rag.git"
```

Optional extras — add what you need:

| Extra | Enables | Pulls in |
|---|---|---|
| `pdf` | PDF ingestion via the vision parser | `pypdfium2`, `pillow` |
| `docintel` | PDF ingestion via Azure Document Intelligence | `azure-ai-documentintelligence` |
| `tokens` | Exact token counting (else a safe heuristic) | `tiktoken` |
| `dev` | Test & lint tooling | `pytest`, `ruff` |

For local development: `git clone … && pip install -e ".[pdf,tokens,dev]"`.

No key yet? `python examples/offline_demo.py` runs the entire pipeline —
chunker, contextualizer, hybrid search, grounded answer — with the endpoint
faked, so you can see the library work (and verify your checkout) before
configuring anything.

## Configure

The package talks to **any OpenAI-compatible endpoint**. Configuration comes
from three places, later ones never overriding earlier ones:

1. **Code** — `configure(...)` (highest precedence)
2. **Environment variables**
3. **A `.env` file** in your working directory (see [.env.example](../.env.example))

```python
from contextual_rag import configure

configure(api_key="sk-...")                                  # standard OpenAI
configure(base_url="https://gateway.example.com/v1")         # corporate gateway
configure(chat_model="gpt-5.4-mini", embed_model="text-embedding-3-large")
```

Importing the package never requires a key — only actually calling an LLM does,
and a missing configuration raises a `ConfigError` that says exactly what to set.

Full reference:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | API key. May be blank for keyless gateways. |
| `OPENAI_BASE_URL` | *(standard OpenAI)* | Any OpenAI-compatible endpoint. |
| `CHAT_MODEL` | `gpt-5.4-mini` | Situating blurbs, LLM rerank, answers. |
| `EMBED_MODEL` | `text-embedding-3-large` | Embeddings. Multilingual-strong recommended. |
| `PARSE_MODEL` | *(falls back to chat)* | Vision model for PDF pages. |
| `CONTEXT_DOC_CAP` | `250000` | Whole-document mode limit (tokens). Keep ≈20k under the chat model's usable input. |
| `CONTEXT_PART_TOKENS` | `48000` | Excerpt size for oversized documents. |
| `CONTEXT_CONCURRENCY` | `4` | Parallel blurb calls per prefix group. `1` = sequential — maximizes cache hits on short documents. |
| `RETRIEVAL_RERANK` | `llm` | Default rerank mode: `llm` \| `endpoint` \| `off`. |
| `RERANK_MODEL` / `RERANK_BASE_URL` / `RERANK_API_KEY` | — | Hosted cross-encoder for `endpoint` mode (Cohere/Jina shape). |
| `VECTOR_DIR` | `./.contextual_rag` | Where the vector store persists. |
| `PARSER` | `vision` | PDF backend: `vision` \| `docintel`. |
| `DOCINTEL_ENDPOINT` / `DOCINTEL_KEY` | — | Azure Document Intelligence credentials. |
| `CHAT_TEMPERATURE` | `0.2` | Default generation temperature. |

## The 5-minute tour

```python
from contextual_rag import ContextualRAG

rag = ContextualRAG("my-corpus")          # a named, persistent collection

# 1. Ingest — each call runs parse → chunk → contextualize → embed → store
rag.ingest_pdf("report.pdf")                              # needs the [pdf] extra
rag.ingest_markdown(open("notes.md").read(), doc_id="notes.md")

# 2. Search — hybrid (dense + BM25 + RRF), reranked by default
hits = rag.search("What drove Q3 margins?", k=8)
for h in hits:
    print(h.score, h.citation, h.section)

# 3. Ask — grounded answer, constrained to retrieved sources, cited inline
ans = rag.ask("What drove Q3 margins?")
print(ans.text)                            # "... margins rose 3pts [S1] ..."
print(ans.sources[0].citation)             # "report.pdf, p.12"
```

The ingestion report tells you what each document cost:

```python
report = rag.ingest_markdown(text, doc_id="doc.md")
# {'doc_id': 'doc.md', 'chunks': 42, 'context_tokens': 61_204,
#  'cached_tokens': 48_130, 'excerpted': False}
```

`cached_tokens` is your verification that prompt-prefix caching is working —
on a multi-page document expect a large fraction of `context_tokens`.

## Recipes

### Debug a retrieval: the per-stage trace

When a chunk you expected doesn't come back, don't guess — look at the stage
that lost it:

```python
trace = rag.search_trace("my question", k=8)
print(trace.timings)                    # {'dense_ms': 42.1, 'bm25_ms': 0.8, ...}
print([c.chunk_id for c in trace.dense][:5])    # what embeddings thought
print([c.chunk_id for c in trace.bm25][:5])     # what keywords thought
print([c.chunk_id for c in trace.fused][:5])    # after RRF
print([c.chunk_id for c in trace.final])        # what the answer will see
top = trace.final[0]
print(top.scores)                       # {'dense': 0.83, 'bm25': 7.1, 'rrf': 0.03, 'rerank': 1.0}
```

### Scope a question to one document

```python
rag.search("what does the contract say about termination?", doc_id="contract.pdf")
rag.ask("what does the contract say about termination?", doc_id="contract.pdf")
```

Both retrievers are scoped — another document's passages cannot leak into the
results or the answer's sources.

### Multiple corpora, one process

```python
finance = ContextualRAG("finance")
legal = ContextualRAG("legal", rerank="off")   # per-instance rerank override
```

Collections are fully isolated (separate Chroma collections, separate caches).

### Resume a bulk ingestion

A document is written in one atomic `add()`, so its presence means it was
fully ingested:

```python
done = rag.doc_ids()
for path in corpus_dir.glob("*.md"):
    if path.name not in done:
        rag.ingest_markdown(path.read_text(), doc_id=path.name)
```

### Keep the blurb model cheap

Blurbs don't need your best model. Route them explicitly:

```python
rag.ingest_markdown(text, doc_id="doc.md", context_model="gpt-5.4-mini")
```

### Use the pipeline as functions (no facade)

Every step is a plain function if you want to own the store or intercept a
stage:

```python
from contextual_rag import (chunk_markdown, contextualize_chunks,
                            VectorStore, search, answer_from)

chunks = chunk_markdown(text, doc_id="doc.md")
ctx = contextualize_chunks(text, chunks)        # one LLM call per chunk
store = VectorStore("corpus")
store.add(ctx)
final = search("my question", store=store, rerank_mode="llm")
print(answer_from("my question", final).text)
```

### Bring your own token counter

The chunker never hard-depends on tiktoken:

```python
chunks = chunk_markdown(text, doc_id="d", count_tokens=lambda t: len(t.split()))
```

### Azure Document Intelligence instead of the vision parser

```bash
pip install "anthropic-contextual-rag[docintel] @ git+..."
```

```dotenv
PARSER=docintel
DOCINTEL_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
DOCINTEL_KEY=...
```

Same `ingest_pdf(...)` call — the backend swap is pure configuration.

## Cost model

| Phase | Cost | When |
|---|---|---|
| Chunking | free (local) | ingestion |
| Contextualization | 1 chat call per chunk; document prefix billed once, then at the cached rate | ingestion (one-time) |
| Embedding | batched calls at ingestion (64 chunks per request) + 1 call per query | ingestion + query |
| BM25 | free (local, pure Python) | query |
| Rerank (`llm`) | exactly 1 chat call per query — skipped automatically when the candidate pool already fits in `k` | query |
| Answer | 1 chat call | query |

Two knobs matter in practice:

- **`CONTEXT_CONCURRENCY`** — parallel blurb calls after the first
  (cache-priming) call of each document. On short documents, `1` gives the
  provider's cache a few seconds to settle between calls and maximizes hits;
  on multi-page documents the default `4` wins on wall-clock.
- **`CONTEXT_DOC_CAP`** — when changing the chat model, keep the cap ≈20k
  tokens under its usable input (250k fits the GPT-5 family; ~100k for a
  128k-context model). Too high = hard API errors mid-ingestion; too low
  merely switches more documents to excerpt mode.
