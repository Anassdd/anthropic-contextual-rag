"""Offline demo — the full pipeline end to end, with NO API key and NO network.

The package has exactly one endpoint seam (``contextual_rag.llm``); this demo
replaces its two functions with tiny deterministic fakes and then runs the
REAL pipeline — chunker, contextualizer, vector store, BM25, RRF, grounded
answer — exactly as production code would:

    python examples/offline_demo.py

Two acts:

1. **The five-line tour** — the ``ContextualRAG`` facade: ingest, search, ask.
2. **Why blurbs matter** — the same corpus indexed twice (with and without
   situating blurbs) and queried head-to-head, a miniature of
   ``examples/benchmark.py``.

It ends with hard assertions: a green run means the pipeline really retrieved
the right chunks and cited the right sources — so this file doubles as an
end-to-end smoke test of your checkout. For a run against a real endpoint see
``quickstart.py``; for measured numbers see ``benchmark.py``.
"""

from __future__ import annotations

import re
import tempfile
import zlib

from contextual_rag import (
    ContextualChunk,
    ContextualRAG,
    VectorStore,
    answer_from,
    chunk_markdown,
    contextualize_chunks,
    llm,
    search_trace,
)

# --------------------------------------------------------------------------- #
# The fake endpoint — deterministic stand-ins for llm.chat / llm.embed.
# Everything else below this section is the real library, unmodified.
# --------------------------------------------------------------------------- #

_DIM = 256
_seen_prefixes: set[str] = set()


class _Result:
    """Shaped like llm.ChatResult, filled with simulated token numbers."""

    def __init__(self, text: str, prompt_chars: int = 0, cached_chars: int = 0):
        self.text = text
        self.model = "fake-offline-model"
        self.usage = type("U", (), {
            "prompt_tokens": prompt_chars // 4,
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": prompt_chars // 4 + max(1, len(text) // 4),
            "cached_tokens": cached_chars // 4,
        })()


def fake_embed(texts, model=None):
    """Deterministic bag-of-words embedding: token -> crc32 bucket (presence).

    Crude, but it gives cosine similarity real meaning (shared vocabulary =
    closeness), which is all the demo needs. Presence rather than counts, so
    repeated stop words ("the", "of") can't drown out the informative terms;
    crc32 rather than hash() so results are identical across runs/machines."""
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for tok in set(re.findall(r"\w+", t.lower())):
            v[zlib.crc32(tok.encode()) % _DIM] = 1.0
        out.append(v if any(v) else [1e-6] * _DIM)
    return out


def fake_chat(messages, **kwargs):
    """Answer the three prompt shapes the pipeline sends.

    - Contextualizer prompt: return a blurb naming the document title and the
      chunk's section — the same information a real LLM extracts.
    - Answer prompt: quote the first sentence of source [S1].
    - Anything else (e.g. a rerank listing): identity ordering.
    """
    content = messages[-1]["content"]

    if "<chunk>" in content:  # contextualizer
        document = content.split(">\n", 1)[1].split("\n</document", 1)[0]
        chunk = content.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        title = next((ln.lstrip("# ").strip() for ln in document.splitlines()
                      if ln.startswith("# ")), "the document")
        section = next((ln.lstrip("# ").strip() for ln in chunk.splitlines()
                        if ln.startswith("## ")), "")
        blurb = f"This chunk is from '{title}'" + (
            f", in the {section} section." if section else ".")
        # Simulate the provider's prompt-prefix cache: the first call carrying a
        # given document pays full price, later ones are ~80% cached.
        prefix = content.split("Here is the chunk", 1)[0]
        cached = int(len(prefix) * 0.8) if prefix in _seen_prefixes else 0
        _seen_prefixes.add(prefix)
        return _Result(blurb, prompt_chars=len(content), cached_chars=cached)

    if "Sources:" in content and "Question:" in content:  # grounded answer
        body = content.split("(source:", 1)[1].split(")\n", 1)[1]
        body = body.split("\n\n[S", 1)[0].split("\n\nQuestion:", 1)[0]
        line = next(ln for ln in body.splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#"))
        sentence = line.split(". ")[0].rstrip(".")
        return _Result(f"{sentence}. [S1]", prompt_chars=len(content))

    return _Result("0,1,2,3,4,5,6,7", prompt_chars=len(content))


llm.chat = fake_chat
llm.embed = fake_embed

# --------------------------------------------------------------------------- #
# A miniature corpus with the ambiguous-chunk failure mode: two status reports
# whose section bodies never name their project — exactly like real documents.
# --------------------------------------------------------------------------- #

NIGHTINGALE = """# Project Nightingale — Status Report

## Overview

The migration progressed steadily this month, with the platform team closing
out the remaining integration work ahead of the review board meeting.

## Budget

Spending is 12 percent under plan for the quarter. The largest saving came
from retiring the legacy cluster earlier than expected, which freed both
licence and support costs.

## Timeline

The rollout is running two weeks ahead of schedule. The final cutover is now
planned for the last week of November, pending sign-off from the review board.
"""

ICARUS = """# Project Icarus — Status Report

## Overview

Integration testing surfaced a compatibility issue with the payments partner,
and the team spent most of the month on remediation and re-testing.

## Budget

Spending is 30 percent over plan for the quarter. The overrun is driven by the
extended contractor engagement and duplicated environments kept alive during
remediation.

## Timeline

The rollout has slipped three weeks behind schedule. A revised cutover date
will be proposed once the partner confirms the fix in their sandbox.
"""

WC = lambda t: len(t.split())  # deterministic word-count "tokens"  # noqa: E731


def act_one(root: str) -> None:
    print("=" * 72)
    print("Act 1 — the five-line tour (ContextualRAG facade)")
    print("=" * 72)

    rag = ContextualRAG("demo", path=root, rerank="off")
    for doc_id, md in (("nightingale.md", NIGHTINGALE), ("icarus.md", ICARUS)):
        report = rag.ingest_markdown(md, doc_id=doc_id)
        print(f"  ingested {doc_id}: {report['chunks']} chunk(s), "
              f"{report['context_tokens']} contextualizer tokens "
              f"({report['cached_tokens']} cached)")
    print(f"  corpus: {rag.count()} chunks from {sorted(rag.doc_ids())}")

    hits = rag.search("What is the status of Project Icarus?", k=2)
    print("\n  search('What is the status of Project Icarus?'):")
    for h in hits:
        print(f"    {h.score:0.4f}  {h.citation}  [{h.section or '—'}]")
        print(f"            blurb: {h.context}")

    ans = rag.ask("What is the budget situation?", doc_id="icarus.md")
    print("\n  ask('What is the budget situation?', doc_id='icarus.md'):")
    print(f"    {ans.text}")
    print(f"    sources: {[s.citation for s in ans.sources]}")
    print("    (the fake LLM simply quotes its top source — a real endpoint "
          "writes a fluent grounded answer)")

    assert hits and hits[0].doc_id == "icarus.md", "search: wrong top document"
    assert "[S1]" in ans.text, "ask: missing citation"
    assert ans.sources and all(s.doc_id == "icarus.md" for s in ans.sources), \
        "ask: doc_id scope leaked another document"


def act_two(root: str) -> None:
    print()
    print("=" * 72)
    print("Act 2 — why blurbs matter (plain vs. contextual, same chunks)")
    print("=" * 72)

    docs = {"nightingale.md": NIGHTINGALE, "icarus.md": ICARUS}
    chunks_by_doc = {d: chunk_markdown(md, doc_id=d, target_tokens=60,
                                       overlap_tokens=0, count_tokens=WC)
                     for d, md in docs.items()}

    plain = VectorStore("plain", path=root)
    contextual = VectorStore("contextual", path=root)
    for doc_id, md in docs.items():
        plain.add([ContextualChunk(chunk=c, context="") for c in chunks_by_doc[doc_id]])
        contextual.add(contextualize_chunks(md, chunks_by_doc[doc_id]))

    example = contextual.all_records()[0]
    print("  what the contextual store indexes (blurb + chunk):")
    print(f"    blurb: {example.context}")
    print(f"    chunk: {example.text.splitlines()[0][:60]}...")

    queries = [
        ("Is Project Nightingale under or over budget?", "nightingale.md", "Budget"),
        ("Is Project Icarus under or over budget?", "icarus.md", "Budget"),
        ("Is the Project Nightingale rollout ahead or behind schedule?",
         "nightingale.md", "Timeline"),
        ("Is the Project Icarus rollout ahead or behind schedule?",
         "icarus.md", "Timeline"),
    ]
    scores = {"plain": 0, "contextual": 0}
    print("\n  top-1 chunk per query (gold = right document AND right section):")
    for query, gold_doc, gold_section in queries:
        print(f"    Q: {query}")
        for label, store in (("plain", plain), ("contextual", contextual)):
            top = search_trace(query, k=3, store=store).final[0]
            ok = top.doc_id == gold_doc and gold_section in top.section
            scores[label] += ok
            print(f"       {label:10s} -> {top.doc_id:16s} [{top.section or '—':8s}] "
                  f"{'✓' if ok else '✗'}")

    print(f"\n  score: plain {scores['plain']}/4, contextual {scores['contextual']}/4")
    print("  The plain store sees 'Spending is 12 percent under plan' with no clue")
    print("  WHOSE spending — the blurb ('This chunk is from Project Nightingale…')")
    print("  restores the signal for both embeddings and BM25.")

    ans = answer_from(queries[0][0],
                      search_trace(queries[0][0], k=2, store=contextual).final)
    print(f"\n  grounded answer: {ans.text}")

    assert scores["contextual"] == 4, "contextual retrieval should ace this corpus"
    assert "[S1]" in ans.text


def main() -> None:
    print(__doc__.split("\n", 1)[0])
    print("(no API key needed — the endpoint seam is faked; everything else is real)\n")
    root = tempfile.mkdtemp(prefix="crag-demo-")
    act_one(root)
    act_two(root)
    print("\n✓ all checks passed — the pipeline works end to end.")


if __name__ == "__main__":
    main()
