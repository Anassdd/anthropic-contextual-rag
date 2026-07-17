"""Contextual Retrieval (Anthropic) — the contextualization step.

For each chunk, an LLM writes a short blurb situating it within its parent document;
that blurb is prepended to the chunk before it gets embedded and BM25-indexed. Anthropic
reported this cuts retrieval failures ~49% (~67% with a reranker). The document is sent
on every chunk call but sits at the START of the prompt, so repeated calls for the same
document hit automatic prompt-prefix caching — which is what makes it cheap.

Documents over CONTEXT_DOC_CAP tokens can't ride whole (a 500k-token SEC filing exceeds
the model's input window, and blurb quality degrades with irrelevant bulk — context rot).
Those are situated against an EXCERPT instead: the document HEAD (title/intro — the
document's identity) plus the REGION around the chunk, rebuilt from the ordered chunk
list. Consecutive chunks are batched — aligned to top-level sections — to share one
byte-identical excerpt, so prefix caching keeps working per batch exactly as it does
per document. Documents at/under the cap keep Anthropic's recipe untouched.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from contextual_rag import llm
from contextual_rag.config import get_settings
from contextual_rag.tokens import count_tokens
from contextual_rag.types import Chunk

# Anthropic's prompt, with the document FIRST so the prefix is cacheable across chunks.
PROMPT_TEMPLATE = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)

# Same instruction, honest framing: the model sees an excerpt, not the whole document.
EXCERPT_TEMPLATE = (
    "<document_excerpt>\n{document}\n</document_excerpt>\n\n"
    "The excerpt above is from a longer document: its beginning, then the part "
    "surrounding the chunk below.\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)

# Excerpt anatomy (tokens). Head = the document's opening (title, TOC, intro) — its
# identity; margins pad the batch on both sides so edge chunks still see surroundings.
_HEAD_TOKENS = 6_000
_MARGIN_TOKENS = 4_000
_MIN_SPAN_TOKENS = 8_000

# Models sometimes ignore "answer only with the context" and add a lead-in like
# "Here is the context:" or wrap the answer in quotes/fences. Strip that — otherwise
# the noise gets embedded and pollutes retrieval.
_PREAMBLE = re.compile(
    r"^\s*(here\s+is\s+(the\s+)?(a\s+)?(succinct\s+|short\s+)?context[^:]*:|context:|"
    r"the\s+context\s+is:?)\s*",
    re.I,
)


def _clean_context(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    if len(t) >= 2 and t[0] in "\"'`" and t[-1] == t[0]:
        t = t[1:-1].strip()
    return _PREAMBLE.sub("", t).strip()


@dataclass
class ContextualChunk:
    """A chunk plus its situating context. `text` is what gets embedded / BM25-indexed."""

    chunk: Chunk
    context: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the prefix cache
    excerpted: bool = False  # situated against an excerpt, not the whole document

    @property
    def text(self) -> str:
        return f"{self.context}\n\n{self.chunk.text}" if self.context else self.chunk.text

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _situate(context_text: str, chunk: Chunk, template: str, model: str | None,
             excerpted: bool = False) -> ContextualChunk:
    messages = [{"role": "user",
                 "content": template.format(document=context_text, chunk=chunk.text)}]
    res = llm.chat(messages, model=model, temperature=0.0)
    usage = res.usage
    return ContextualChunk(
        chunk=chunk,
        context=_clean_context(res.text),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cached_tokens=getattr(usage, "cached_tokens", 0) if usage else 0,
        excerpted=excerpted,
    )


def contextualize_chunk(document_markdown: str, chunk: Chunk, *, model: str | None = None) -> ContextualChunk:
    """One LLM call: whole document (cacheable prefix) + the chunk -> a short context blurb."""
    return _situate(document_markdown, chunk, PROMPT_TEMPLATE, model)


def _situate_group(context_text: str, chunks: list[Chunk], template: str,
                   model: str | None, excerpted: bool, workers: int) -> list[ContextualChunk]:
    """All chunks sharing one prompt prefix. The FIRST call runs alone — it writes the
    prefix cache — then the rest run in a small thread pool, reading it. Parallelizing
    the first call too would make every worker pay the full uncached prefix."""
    first = _situate(context_text, chunks[0], template, model, excerpted)
    rest = chunks[1:]
    if workers <= 1 or len(rest) <= 1:
        return [first] + [_situate(context_text, c, template, model, excerpted) for c in rest]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [first] + list(pool.map(
            lambda c: _situate(context_text, c, template, model, excerpted), rest))


def contextualize_chunks(document_markdown: str, chunks, *, model: str | None = None,
                         doc_cap: int | None = None, part_tokens: int | None = None,
                         concurrency: int | None = None) -> list[ContextualChunk]:
    """Contextualize every chunk of one document. Whole-document mode when the document
    fits `doc_cap`; excerpt mode otherwise. `concurrency` workers per prefix group
    (default CONTEXT_CONCURRENCY; 1 = fully sequential)."""
    chunks = list(chunks)
    if not chunks:
        return []
    settings = get_settings()
    workers = concurrency if concurrency is not None else settings.context_concurrency
    cap = doc_cap if doc_cap is not None else settings.context_doc_cap
    if count_tokens(document_markdown) <= cap:
        return _situate_group(document_markdown, chunks, PROMPT_TEMPLATE, model, False, workers)
    budget = part_tokens if part_tokens is not None else settings.context_part_tokens
    return _contextualize_excerpted(chunks, model, budget, workers)


# ---- excerpt mode ----------------------------------------------------------
# The ordered chunk list IS the document (the chunker walks it top-to-bottom and keeps
# every character), so regions are found by position in that list — no offset mapping,
# no re-tokenizing: each Chunk already carries its token_count.


def _section(chunk: Chunk) -> str:
    return chunk.header_path[0] if chunk.header_path else ""


def _chunk_tokens(chunks: list[Chunk]) -> list[int]:
    return [c.token_count or count_tokens(c.text) for c in chunks]


def _head_end(tok: list[int]) -> int:
    """Index right after the chunks forming the document head (~_HEAD_TOKENS)."""
    used = 0
    for i, t in enumerate(tok):
        used += t
        if used >= _HEAD_TOKENS:
            return i + 1
    return len(tok)


def _batch_ranges(chunks: list[Chunk], tok: list[int], span_budget: int) -> list[tuple[int, int]]:
    """Consecutive [start, end) ranges whose chunks share one excerpt. A range closes
    when it would outgrow the span budget — or, once at least half full, at a top-level
    section boundary, so an excerpt is a coherent region rather than an arbitrary cut."""
    ranges, start, used = [], 0, tok[0]
    for i in range(1, len(chunks)):
        new_section = _section(chunks[i]) != _section(chunks[i - 1])
        if used + tok[i] > span_budget or (new_section and used >= span_budget // 2):
            ranges.append((start, i))
            start, used = i, 0
        used += tok[i]
    ranges.append((start, len(chunks)))
    return ranges


def _excerpt_for(chunks: list[Chunk], tok: list[int], head_end: int,
                 start: int, end: int) -> str:
    """head + [gap marker] + the batch's region widened by margins on both sides.
    The region never re-includes head chunks, so nothing repeats."""
    lo, hi = max(head_end, start), max(head_end, end)
    need = _MARGIN_TOKENS
    while lo > head_end and need > 0:
        lo -= 1
        need -= tok[lo]
    need = _MARGIN_TOKENS
    while hi < len(chunks) and need > 0:
        need -= tok[hi]
        hi += 1
    parts = [c.text for c in chunks[:head_end]]
    if lo > head_end:
        parts.append("[…]")
    parts.extend(c.text for c in chunks[lo:hi])
    return "\n\n".join(parts)


def _contextualize_excerpted(chunks: list[Chunk], model: str | None,
                             part_budget: int, workers: int) -> list[ContextualChunk]:
    tok = _chunk_tokens(chunks)
    head_end = _head_end(tok)
    span = max(_MIN_SPAN_TOKENS, part_budget - _HEAD_TOKENS - 2 * _MARGIN_TOKENS)
    out: list[ContextualChunk] = []
    for start, end in _batch_ranges(chunks, tok, span):
        excerpt = _excerpt_for(chunks, tok, head_end, start, end)
        out.extend(_situate_group(excerpt, chunks[start:end], EXCERPT_TEMPLATE,
                                  model, True, workers))
    return out
