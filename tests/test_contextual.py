"""Contextualizer tests — offline, no LLM calls (chat is faked via monkeypatch).

Verifies the prompt structure, how the context blurb is assembled onto the chunk,
excerpt mode for oversized documents, and cache-aware concurrency.
"""

import threading

from contextual_rag import (  # noqa: F401 (llm patched via monkeypatch)
    chunk_markdown,
    contextual,
    llm,
)

DOC = "# Intro\n\nThis paper studies Hölder inequalities.\n\n## Methods\n\nWe use convexity."


class _Usage:
    prompt_tokens, completion_tokens, total_tokens, cached_tokens = 120, 12, 132, 64


class _Result:
    usage = _Usage()
    model = "fake-model"

    def __init__(self, text):
        self.text = text


def test_prompt_contains_document_and_chunk(monkeypatch):
    captured = {}

    def fake(messages, **kw):
        captured["content"] = messages[0]["content"]
        captured["kw"] = kw
        return _Result("From the Methods section.")

    monkeypatch.setattr(contextual.llm, "chat", fake)
    chunks = chunk_markdown(DOC, doc_id="d")
    contextual.contextualize_chunk(DOC, chunks[-1])
    c = captured["content"]
    assert "<document>" in c and "</document>" in c, "document not wrapped"
    assert "<chunk>" in c and "</chunk>" in c, "chunk not wrapped"
    assert "Hölder" in c, "document body missing from prompt"
    assert c.index("<document>") < c.index("<chunk>"), "document must come first (cacheable prefix)"
    assert captured["kw"].get("temperature") == 0.0, "should request temperature 0"


def test_assembles_context_onto_chunk(monkeypatch):
    monkeypatch.setattr(contextual.llm, "chat",
                        lambda messages, **kw: _Result("  This chunk is from the intro.  "))
    chunks = chunk_markdown(DOC, doc_id="d")
    out = contextual.contextualize_chunks(DOC, chunks)
    assert len(out) == len(chunks)
    cc = out[0]
    assert cc.context == "This chunk is from the intro.", "context not stripped/captured"
    assert cc.text.startswith("This chunk is from the intro."), "context not prepended"
    assert cc.chunk.text in cc.text, "original chunk text must be preserved in the payload"
    assert cc.prompt_tokens == 120 and cc.completion_tokens == 12, "token accounting wrong"
    assert cc.total_tokens == 132


def test_empty_context_is_safe(monkeypatch):
    monkeypatch.setattr(contextual.llm, "chat", lambda messages, **kw: _Result(""))
    out = contextual.contextualize_chunks(DOC, chunk_markdown(DOC, doc_id="d"))
    # with no context, text falls back to the bare chunk (no leading blank lines)
    assert out[0].text == out[0].chunk.text, "empty context should leave the chunk unchanged"


def test_no_chunks(monkeypatch):
    monkeypatch.setattr(contextual.llm, "chat", lambda *a, **k: _Result("x"))
    assert contextual.contextualize_chunks(DOC, []) == []


def test_cached_tokens_passthrough(monkeypatch):
    monkeypatch.setattr(contextual.llm, "chat", lambda messages, **kw: _Result("Intro context."))
    out = contextual.contextualize_chunks(DOC, chunk_markdown(DOC, doc_id="d"))
    assert out[0].cached_tokens == 64, "cached_tokens not carried from usage"


# ---- where it can go wrong: the LLM ignores 'context only' and adds noise ----

def test_clean_strips_preamble():
    for raw, want in [
        ("Here is the succinct context: This is the intro.", "This is the intro."),
        ("Here is a short context for the chunk: From methods.", "From methods."),
        ("Context: Belongs to results.", "Belongs to results."),
        ("The context is: Section two.", "Section two."),
    ]:
        got = contextual._clean_context(raw)
        assert got == want, f"preamble not stripped: {raw!r} -> {got!r}"


def test_clean_strips_quotes_and_fences():
    assert contextual._clean_context('"From the intro."') == "From the intro."
    assert contextual._clean_context("'From the intro.'") == "From the intro."
    assert contextual._clean_context("```\nFrom the intro.\n```") == "From the intro."


def test_clean_preserves_real_context_and_french():
    clean = "This chunk introduces Hölder inequalities in the opening section."
    assert contextual._clean_context(clean) == clean, "clean context must be untouched"
    fr = "Le chunk se situe dans la section sur l'inégalité de Hölder."
    assert contextual._clean_context(fr) == fr, "French context must be preserved verbatim"


# ---- excerpt mode: documents too big to ride whole -------------------------

def _big_doc(sections=("Alpha", "Beta", "Gamma", "Delta"), paras=8):
    parts = []
    for s in sections:
        parts.append(f"# {s} section")
        parts.extend(f"In {s}, paragraph {i} explains the {s.lower()} topic number {i} "
                     "in enough words to carry a few dozen tokens of body text."
                     for i in range(paras))
    return "\n\n".join(parts)


def _run_excerpted(monkeypatch, doc, chunks, part_tokens=340):
    """contextualize with a tiny cap (forcing excerpt mode), capturing every prompt.
    Shrinks the excerpt anatomy so a page-sized fixture exercises batching."""
    prompts = []

    def fake(messages, **kw):
        prompts.append(messages[0]["content"])
        return _Result("A situating blurb.")

    monkeypatch.setattr(contextual.llm, "chat", fake)
    monkeypatch.setattr(contextual, "_HEAD_TOKENS", 60)
    monkeypatch.setattr(contextual, "_MARGIN_TOKENS", 40)
    monkeypatch.setattr(contextual, "_MIN_SPAN_TOKENS", 50)
    out = contextual.contextualize_chunks(doc, chunks, doc_cap=50, part_tokens=part_tokens)
    return out, prompts


def _prefix(prompt):
    return prompt.split("Here is the chunk", 1)[0]


def test_cap_switches_modes(monkeypatch):
    from contextual_rag.tokens import count_tokens

    doc = _big_doc()
    chunks = chunk_markdown(doc, doc_id="d", target_tokens=60)
    prompts = []

    def fake(messages, **kw):
        prompts.append(messages[0]["content"])
        return _Result("Blurb.")

    monkeypatch.setattr(contextual.llm, "chat", fake)
    whole = contextual.contextualize_chunks(doc, chunks, doc_cap=count_tokens(doc))
    assert all("<document>" in p and "<document_excerpt>" not in p for p in prompts)
    assert not any(c.excerpted for c in whole), "under the cap must stay whole-doc mode"

    out, prompts = _run_excerpted(monkeypatch, doc, chunks)
    assert all("<document_excerpt>" in p for p in prompts)
    assert all(c.excerpted for c in out), "over the cap must mark chunks excerpted"


def test_excerpt_batches_share_prefix_and_cover_chunks(monkeypatch):
    doc = _big_doc()
    chunks = chunk_markdown(doc, doc_id="d", target_tokens=60)
    out, prompts = _run_excerpted(monkeypatch, doc, chunks)
    assert len(out) == len(chunks)
    assert all(cc.chunk is c for cc, c in zip(out, chunks, strict=True))
    prefixes = [_prefix(p) for p in prompts]
    distinct = list(dict.fromkeys(prefixes))
    assert 1 < len(distinct) < len(chunks), f"expected several shared batches, got {len(distinct)}"
    for i, p in enumerate(prompts):  # a chunk must sit inside its own excerpt
        assert chunks[i].text in _prefix(p), f"chunk {i} missing from its excerpt"
    assert prefixes == sorted(prefixes, key=distinct.index), "batches must be consecutive"


def test_excerpt_head_and_sections(monkeypatch):
    doc = _big_doc()
    chunks = chunk_markdown(doc, doc_id="d", target_tokens=60)
    out, prompts = _run_excerpted(monkeypatch, doc, chunks)
    head = chunks[0].text
    assert all(head in _prefix(p) for p in prompts), "document head must open every excerpt"
    by_prefix = {}
    for i, p in enumerate(prompts):
        by_prefix.setdefault(_prefix(p), []).append(chunks[i].header_path[0])
    mixed = [set(secs) for secs in by_prefix.values() if len(set(secs)) > 1]
    assert not mixed, f"batches must not straddle sections: {mixed}"


# ---- concurrency: prime the cache first, then parallel workers -------------

def test_concurrent_prime_first_and_order(monkeypatch):
    from contextual_rag.tokens import count_tokens

    doc = _big_doc()
    chunks = chunk_markdown(doc, doc_id="d", target_tokens=60)
    events, lock = [], threading.Lock()

    def fake(messages, **kw):
        chunk_body = messages[0]["content"].split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        with lock:
            events.append(("start", chunk_body))
        with lock:
            events.append(("end", chunk_body))
        return _Result("blurb")

    monkeypatch.setattr(contextual.llm, "chat", fake)
    out = contextual.contextualize_chunks(doc, chunks, doc_cap=count_tokens(doc),
                                          concurrency=4)
    assert [cc.chunk for cc in out] == chunks, "results must keep chunk order"
    assert events[0] == ("start", chunks[0].text) and events[1] == ("end", chunks[0].text), \
        "the first (cache-priming) call must run alone before any worker starts"
    assert len(events) == 2 * len(chunks)


def test_concurrency_one_is_sequential(monkeypatch):
    doc = _big_doc()
    chunks = chunk_markdown(doc, doc_id="d", target_tokens=60)
    order = []
    monkeypatch.setattr(contextual.llm, "chat",
                        lambda messages, **kw: (order.append(1), _Result("b"))[1])
    out = contextual.contextualize_chunks(doc, chunks, doc_cap=10**9, concurrency=1)
    assert len(out) == len(chunks) == len(order)
