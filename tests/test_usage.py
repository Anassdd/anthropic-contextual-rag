"""Usage-pattern tests — the documented API promises, exercised offline.

Each test mirrors something README.md / docs/usage.md tells a new user to do;
if one of these fails, the documentation is lying. The endpoint seam is faked
(bag-of-words embeddings, templated blurbs and answers), so everything runs
with no key and no network — the same trick examples/offline_demo.py uses.
"""

import re
import tempfile
import zlib

import pytest

from contextual_rag import (
    ContextualRAG,
    VectorStore,
    answer_from,
    chunk_markdown,
    contextualize_chunks,
    llm,
    search,
)

DOC_A = """# Project Nightingale — Status Report

## Budget

Spending is 12 percent under plan for the quarter thanks to early cluster retirement.

## Timeline

The rollout is running two weeks ahead of schedule going into November.
"""

DOC_B = """# Project Icarus — Status Report

## Budget

Spending is 30 percent over plan for the quarter due to the extended remediation.

## Timeline

The rollout has slipped three weeks behind schedule pending the partner fix.
"""


def _fake_embed(texts, model=None):
    """Bag-of-words presence vectors (crc32 buckets): deterministic, and cosine
    similarity means 'shares vocabulary' — enough for retrieval to be real."""
    out = []
    for t in texts:
        v = [0.0] * 128
        for tok in set(re.findall(r"\w+", t.lower())):
            v[zlib.crc32(tok.encode()) % 128] = 1.0
        out.append(v if any(v) else [1e-6] * 128)
    return out


def _fake_chat(messages, **kwargs):
    content = messages[-1]["content"]

    class R:
        usage = None
        model = "fake"
        text = ""

    if "<chunk>" in content:  # contextualizer prompt -> blurb with title+section
        document = content.split(">\n", 1)[1].split("\n</document", 1)[0]
        chunk = content.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        title = next((ln.lstrip("# ").strip() for ln in document.splitlines()
                      if ln.startswith("# ")), "the document")
        section = next((ln.lstrip("# ").strip() for ln in chunk.splitlines()
                        if ln.startswith("## ")), "")
        R.text = f"This chunk is from '{title}'" + (
            f", in the {section} section." if section else ".")
    elif "Sources:" in content and "Question:" in content:  # answer prompt
        R.text = "Grounded answer quoting the sources [S1]."
    else:  # rerank listing -> identity order
        R.text = "0,1,2,3,4,5,6,7"
    return R()


@pytest.fixture(autouse=True)
def _fake_endpoint(monkeypatch):
    monkeypatch.setattr(llm, "embed", _fake_embed)
    monkeypatch.setattr(llm, "chat", _fake_chat)


@pytest.fixture()
def root():
    return tempfile.mkdtemp(prefix="crag-usage-")


def _ingest_both(rag):
    rag.ingest_markdown(DOC_A, doc_id="nightingale.md")
    rag.ingest_markdown(DOC_B, doc_id="icarus.md")


def test_facade_five_minute_tour(root):
    """The README quickstart, line by line: ingest -> search -> ask."""
    rag = ContextualRAG("tour", path=root, rerank="off")

    report = rag.ingest_markdown(DOC_A, doc_id="nightingale.md")
    for key in ("doc_id", "chunks", "context_tokens", "cached_tokens", "excerpted"):
        assert key in report, f"ingestion report missing {key!r}"
    assert report["chunks"] >= 1 and rag.count() == report["chunks"]

    rag.ingest_markdown(DOC_B, doc_id="icarus.md")
    assert rag.doc_ids() == {"nightingale.md", "icarus.md"}

    hits = rag.search("Is Project Icarus over budget?", k=3)
    assert hits, "search returned nothing"
    top = hits[0]
    assert top.doc_id == "icarus.md", f"wrong top doc: {top.doc_id}"
    assert top.citation.startswith("icarus.md"), "citation must name the document"
    assert top.context, "contextual blurb missing from the stored chunk"
    assert top.score > 0 and "rrf" in top.scores

    ans = rag.ask("Is Project Icarus over budget?", k=2)
    assert "[S1]" in ans.text
    assert ans.sources and ans.sources[0].doc_id == "icarus.md"


def test_ask_scoped_to_one_document(root):
    """docs/usage.md: doc_id scopes retrieval — other documents can't leak in."""
    rag = ContextualRAG("scoped", path=root, rerank="off")
    _ingest_both(rag)
    ans = rag.ask("What is the budget situation?", doc_id="nightingale.md")
    assert ans.sources, "scoped ask found no sources"
    assert all(s.doc_id == "nightingale.md" for s in ans.sources), \
        "doc_id scope leaked another document into the answer"


def test_resume_pattern_is_idempotent(root):
    """docs/usage.md 'resume a bulk ingestion': presence in doc_ids() means
    fully ingested, and re-ingesting a document must not duplicate chunks."""
    rag = ContextualRAG("resume", path=root, rerank="off")
    _ingest_both(rag)
    count_before = rag.count()

    done = rag.doc_ids()
    for doc_id, md in (("nightingale.md", DOC_A), ("icarus.md", DOC_B)):
        if doc_id not in done:  # the documented skip loop — nothing to do here
            rag.ingest_markdown(md, doc_id=doc_id)
    assert rag.count() == count_before

    rag.ingest_markdown(DOC_A, doc_id="nightingale.md")  # forced re-ingest
    assert rag.count() == count_before, "re-ingesting a doc must upsert, not duplicate"


def test_collections_are_isolated(root):
    """docs/usage.md 'multiple corpora': same path, separate collections."""
    finance = ContextualRAG("finance", path=root, rerank="off")
    legal = ContextualRAG("legal", path=root, rerank="off")
    finance.ingest_markdown(DOC_A, doc_id="nightingale.md")
    legal.ingest_markdown(DOC_B, doc_id="icarus.md")

    assert finance.doc_ids() == {"nightingale.md"}
    assert legal.doc_ids() == {"icarus.md"}
    hits = finance.search("Is the rollout behind schedule?", k=5)
    assert all(h.doc_id == "nightingale.md" for h in hits), \
        "a collection surfaced chunks from another collection"


def test_functional_pipeline_no_facade(root):
    """docs/usage.md 'use the pipeline as functions': every stage is a plain
    function and composes exactly as documented."""
    chunks = chunk_markdown(DOC_B, doc_id="icarus.md")
    ctx = contextualize_chunks(DOC_B, chunks)
    assert len(ctx) == len(chunks)
    assert all(cc.context for cc in ctx), "every chunk should carry a blurb"

    store = VectorStore("functional", path=root)
    assert store.add(ctx) == len(ctx)

    final = search("Is the project over budget?", store=store, rerank_mode="off")
    assert final and final[0].doc_id == "icarus.md"

    ans = answer_from("Is the project over budget?", final)
    assert "[S1]" in ans.text and ans.sources == final


def test_ingest_pdf_bytes_requires_filename(root):
    """Bytes with no filename would silently collide on a default doc_id —
    the facade must refuse instead."""
    rag = ContextualRAG("pdfbytes", path=root, rerank="off")
    with pytest.raises(ValueError, match="filename"):
        rag.ingest_pdf(b"%PDF-1.4 not a real pdf")
