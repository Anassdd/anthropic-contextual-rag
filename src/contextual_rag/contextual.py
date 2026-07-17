"""Contextual Retrieval — the contextualization step (the heart of the package).

For each chunk, an LLM writes a short blurb situating it within its parent
document; that blurb is prepended to the chunk before embedding and BM25
indexing. Anthropic, who published the technique, reported it cuts retrieval
failures by ~49% (~67% with a reranker):
https://www.anthropic.com/news/contextual-retrieval

**Why it's affordable** — the document is sent on every chunk call but sits at
the *start* of the prompt, byte-identical across calls, so repeated calls hit
the provider's automatic prompt-prefix cache: the document is billed in full
once, then at the cached rate (typically ~10%) for every other chunk.
:class:`ContextualChunk.cached_tokens` surfaces the cache hits so callers can
verify it is working.

**Oversized documents — excerpt mode.** The published recipe assumes the
document fits the model's input window; a 500k-token SEC filing doesn't (and
blurb quality degrades with irrelevant bulk — "context rot"). Documents over
the cap are situated against an *excerpt* instead: the document HEAD
(title/TOC/intro — its identity) plus the REGION around the chunk, rebuilt from
the ordered chunk list. Consecutive chunks are batched — aligned to top-level
section boundaries — to share one byte-identical excerpt, so prefix caching
keeps working per batch exactly as it does per document. Documents at or under
the cap get the published recipe untouched.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from contextual_rag import llm
from contextual_rag.config import get_settings
from contextual_rag.tokens import count_tokens
from contextual_rag.types import Chunk

#: Anthropic's published prompt, with the document FIRST so the prefix is
#: byte-identical — and therefore cacheable — across all chunks of a document.
PROMPT_TEMPLATE = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)

#: Excerpt-mode variant: same situating instruction, honest framing — the model
#: is told it sees an excerpt, not the whole document.
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

# Excerpt anatomy (tokens). Head = the document's opening (title, TOC, intro) —
# its identity; margins pad the batch on both sides so edge chunks still see
# their surroundings.
_HEAD_TOKENS = 6_000
_MARGIN_TOKENS = 4_000
_MIN_SPAN_TOKENS = 8_000

# Models sometimes ignore "answer only with the context" and add a lead-in like
# "Here is the context:" or wrap the answer in quotes/fences. Strip that —
# otherwise the noise gets embedded and pollutes retrieval.
_PREAMBLE = re.compile(
    r"^\s*(here\s+is\s+(the\s+)?(a\s+)?(succinct\s+|short\s+)?context[^:]*:|context:|"
    r"the\s+context\s+is:?)\s*",
    re.I,
)


def _clean_context(text: str) -> str:
    """Normalize a raw blurb: drop wrapping fences/quotes, then any lead-in."""
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    if len(t) >= 2 and t[0] in "\"'`" and t[-1] == t[0]:
        t = t[1:-1].strip()
    return _PREAMBLE.sub("", t).strip()


@dataclass
class ContextualChunk:
    """A chunk plus its situating context — the contextualizer's output.

    Attributes:
        chunk: The original :class:`~contextual_rag.types.Chunk`, provenance
            intact.
        context: The LLM's situating blurb (may be empty if the model
            returned nothing usable).
        prompt_tokens: Tokens sent for this chunk's blurb call.
        completion_tokens: Tokens the blurb itself cost.
        cached_tokens: Prompt tokens served from the prefix cache — the
            cost-saving mechanism, surfaced for verification.
        excerpted: True if situated against an excerpt (oversized-document
            mode) rather than the whole document.
    """

    chunk: Chunk
    context: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    excerpted: bool = False

    @property
    def text(self) -> str:
        """The embed/BM25 payload: blurb + chunk (bare chunk if no blurb)."""
        return f"{self.context}\n\n{self.chunk.text}" if self.context else self.chunk.text

    @property
    def total_tokens(self) -> int:
        """Total tokens this chunk's contextualization cost."""
        return self.prompt_tokens + self.completion_tokens


def _situate(context_text: str, chunk: Chunk, template: str, model: str | None,
             excerpted: bool = False) -> ContextualChunk:
    """One blurb call: (document or excerpt) + chunk → cleaned ContextualChunk."""
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


def contextualize_chunk(document_markdown: str, chunk: Chunk,
                        *, model: str | None = None) -> ContextualChunk:
    """Situate a single chunk against its whole document (one LLM call).

    Args:
        document_markdown: The full parent document (the cacheable prefix).
        chunk: The chunk to situate.
        model: Override the configured chat model for this call.

    Returns:
        The chunk with its blurb and token accounting.
    """
    return _situate(document_markdown, chunk, PROMPT_TEMPLATE, model)


def _situate_group(context_text: str, chunks: list[Chunk], template: str,
                   model: str | None, excerpted: bool, workers: int) -> list[ContextualChunk]:
    """Contextualize all chunks sharing one prompt prefix, cache-safely.

    The FIRST call runs alone — it writes the prefix cache — then the rest run
    in a small thread pool, reading it. Parallelizing the first call too would
    make every worker pay the full uncached prefix price simultaneously."""
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
    """Contextualize every chunk of one document (the main entry point).

    Chooses whole-document mode (the published recipe) when the document fits
    ``doc_cap`` tokens, excerpt mode otherwise — automatically, per document.

    Args:
        document_markdown: The full parent document.
        chunks: The document's chunks, in reading order (the order matters:
            excerpt mode rebuilds document regions from it).
        model: Override the configured chat model (a cheap tier is fine for
            blurbs).
        doc_cap: Whole-document mode limit in tokens. Default:
            ``Settings.context_doc_cap``.
        part_tokens: Excerpt size target in excerpt mode. Default:
            ``Settings.context_part_tokens``.
        concurrency: Parallel calls per prompt-prefix group; 1 = fully
            sequential. Default: ``Settings.context_concurrency``.

    Returns:
        One :class:`ContextualChunk` per input chunk, same order.
    """
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


# --------------------------------------------------------------------------- #
# Excerpt mode — documents too large to ride whole
# --------------------------------------------------------------------------- #
# The ordered chunk list IS the document (the chunker walks it top-to-bottom
# and keeps every character), so regions are found by position in that list —
# no offset mapping, no re-tokenizing: each Chunk already carries token_count.


def _section(chunk: Chunk) -> str:
    """The chunk's top-level section (batch boundaries snap to these)."""
    return chunk.header_path[0] if chunk.header_path else ""


def _chunk_tokens(chunks: list[Chunk]) -> list[int]:
    return [c.token_count or count_tokens(c.text) for c in chunks]


def _head_end(tok: list[int], head_tokens: int) -> int:
    """Index right after the chunks forming the document head (~head_tokens)."""
    used = 0
    for i, t in enumerate(tok):
        used += t
        if used >= head_tokens:
            return i + 1
    return len(tok)


def _batch_ranges(chunks: list[Chunk], tok: list[int], span_budget: int) -> list[tuple[int, int]]:
    """Consecutive ``[start, end)`` ranges whose chunks share one excerpt.

    A range closes when it would outgrow the span budget — or, once at least
    half full, at a top-level section boundary, so an excerpt is a coherent
    region of the document rather than an arbitrary cut."""
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
                 start: int, end: int, margin_tokens: int) -> str:
    """Assemble one excerpt: head + [gap marker] + the batch's region widened
    by margins on both sides. The region never re-includes head chunks, so
    nothing repeats."""
    lo, hi = max(head_end, start), max(head_end, end)
    need = margin_tokens
    while lo > head_end and need > 0:
        lo -= 1
        need -= tok[lo]
    need = margin_tokens
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
    """Excerpt-mode driver: batch, build each shared excerpt, situate per batch.

    The default excerpt anatomy (~6k head + 2×4k margins + ≥8k span) floors at
    ~22k tokens; when `part_budget` asks for less — a small-context model —
    the anatomy is scaled down proportionally so the budget is actually
    honored instead of silently overshot."""
    head_t, margin_t, min_span = _HEAD_TOKENS, _MARGIN_TOKENS, _MIN_SPAN_TOKENS
    floor = head_t + 2 * margin_t + min_span
    if part_budget < floor:
        scale = part_budget / floor
        head_t = max(1, int(head_t * scale))
        margin_t = max(1, int(margin_t * scale))
        min_span = max(1, int(min_span * scale))
    tok = _chunk_tokens(chunks)
    head_end = _head_end(tok, head_t)
    span = max(min_span, part_budget - head_t - 2 * margin_t)
    out: list[ContextualChunk] = []
    for start, end in _batch_ranges(chunks, tok, span):
        excerpt = _excerpt_for(chunks, tok, head_end, start, end, margin_t)
        out.extend(_situate_group(excerpt, chunks[start:end], EXCERPT_TEMPLATE,
                                  model, True, workers))
    return out
