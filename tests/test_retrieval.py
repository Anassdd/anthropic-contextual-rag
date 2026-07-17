"""Retrieval-engine tests — offline, no real embeddings/LLM (both faked).

Covers the store (add/count/get/query), dense search, BM25 exact-term, RRF fusion via
search_trace, the grounded answer assembly, and the ContextualRAG facade. Uses a
deterministic bag-of-words embedder so dense similarity is meaningful and reproducible.
"""

import re
import tempfile

import pytest

from contextual_rag import VectorStore, answer_from, llm, search_trace
from contextual_rag.bm25 import BM25
from contextual_rag.contextual import ContextualChunk
from contextual_rag.types import Chunk

VOCAB = ["hölder", "inequality", "exponent", "exponents", "conjugate", "convexity",
         "results", "bound", "cauchy", "schwarz", "française", "prompt", "caching",
         "retrieval", "cheap", "généralise"]


def fake_embed(texts, model=None):
    vecs = []
    for t in texts:
        toks = set(re.findall(r"\w+", t.lower()))
        v = [1.0 if w in toks else 0.0 for w in VOCAB]
        vecs.append(v if any(v) else [1e-6] * len(VOCAB))
    return vecs


# Every module does `from contextual_rag import llm` and calls llm.embed/llm.chat at
# call time, so patching the shared module object fakes the whole package at once.
@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    monkeypatch.setattr(llm, "embed", fake_embed)


def _cc(i, doc, text, context, pages, section):
    ch = Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, index=i, text=text,
               header_path=section.split(" › ") if section else [], pages=pages)
    return ContextualChunk(chunk=ch, context=context)


def build_store():
    store = VectorStore("default", path=tempfile.mkdtemp())
    store.add([
        _cc(0, "math.pdf", "Hölder inequality requires conjugate exponents p and q.",
            "Methods section defining the exponents.", [2], "Methods"),
        _cc(1, "math.pdf", "The results confirm the bound is tight via a convexity argument.",
            "Results section.", [4], "Results"),
        _cc(2, "french.pdf", "L'inégalité de Hölder généralise Cauchy-Schwarz.",
            "Section française d'introduction.", [1], "Introduction"),
        _cc(3, "rag.pdf", "Prompt caching makes contextual retrieval cheap.",
            "Caching section.", [3], "Caching"),
    ])
    return store


def test_store_roundtrip():
    s = build_store()
    assert s.count() == 4, f"expected 4, got {s.count()}"
    got = s.get("math.pdf::0")
    assert got and "conjugate exponents" in got.text
    assert got.pages == [2] and got.section == "Methods"
    assert s.doc_ids() == {"math.pdf", "french.pdf", "rag.pdf"}


def test_dense_query():
    s = build_store()
    qvec = fake_embed(["Hölder conjugate exponents"])[0]
    hits = s.query(qvec, 3)
    assert hits[0].chunk_id == "math.pdf::0", f"dense top wrong: {hits[0].chunk_id}"
    assert hits[0].scores["dense"] > 0


def test_bm25_exact_term():
    s = build_store()
    recs = s.all_records()
    bm = BM25().build([(r.chunk_id, r.embed_text) for r in recs])
    hits = bm.search("convexity", 3)  # rare term, only in chunk 1
    assert hits and hits[0][0] == "math.pdf::1", f"bm25 exact-term wrong: {hits}"


def test_search_trace_fuses():
    s = build_store()
    tr = search_trace("Hölder exponents", k=4, store=s)
    assert tr.dense and tr.bm25 and tr.fused and tr.final, "a stage came back empty"
    top_ids = [c.chunk_id for c in tr.final[:2]]
    assert "math.pdf::0" in top_ids, f"expected math.pdf::0 in top-2, got {top_ids}"
    # the top fused chunk carries scores from both retrievers
    assert "rrf" in tr.fused[0].scores, "fused chunk missing rrf score"


def test_rrf_rewards_agreement():
    s = build_store()
    tr = search_trace("Hölder conjugate exponents", k=4, store=s)
    # math.pdf::0 is found by BOTH dense and bm25 -> should top the fused list
    assert tr.fused[0].chunk_id == "math.pdf::0", f"RRF winner wrong: {tr.fused[0].chunk_id}"


def test_doc_scoped_search():
    s = build_store()
    tr = search_trace("Hölder", k=4, store=s, doc_id="french.pdf")
    assert tr.final, "doc-scoped search came back empty"
    assert all(c.doc_id == "french.pdf" for c in tr.final), \
        "doc_id scope leaked another document's chunks"


def test_answer_from_grounded(monkeypatch):
    captured = {}

    def fake_chat(messages, **kw):
        captured["messages"] = messages

        class R:
            text = "Hölder requires conjugate exponents p, q [S1]."
            usage = None
        return R()

    monkeypatch.setattr(llm, "chat", fake_chat)
    s = build_store()
    chunks = search_trace("Hölder exponents", k=2, store=s).final
    ans = answer_from("What exponents does Hölder require?", chunks)
    assert ans.sources, "answer has no sources"
    user_msg = captured["messages"][-1]["content"]
    assert "[S1]" in user_msg and "source:" in user_msg, "grounded prompt missing source labels"
    assert "[S1]" in ans.text


def test_facade_roundtrip(monkeypatch):
    from contextual_rag import ContextualRAG

    class R:
        text = "Situating blurb."
        usage = None
        model = "fake"

    monkeypatch.setattr(llm, "chat", lambda messages, **kw: R())
    rag = ContextualRAG("facade-test", path=tempfile.mkdtemp(), rerank="off")
    report = rag.ingest_markdown(
        "# Doc\n\nPrompt caching makes contextual retrieval cheap.", doc_id="doc.md")
    assert report["chunks"] >= 1 and rag.count() == report["chunks"]
    hits = rag.search("prompt caching", k=2)
    assert hits and hits[0].doc_id == "doc.md"
    rag.reset()
    assert rag.count() == 0
