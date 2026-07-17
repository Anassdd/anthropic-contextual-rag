"""Contextual vs. plain retrieval — a measurable A/B benchmark.

Reproduces, at demo scale, the evaluation behind Anthropic's Contextual
Retrieval results (https://www.anthropic.com/news/contextual-retrieval): build
the SAME corpus twice — once with bare chunks, once with contextualized
chunks — and measure Pass@k (is a gold chunk in the top-k?) on questions whose
answers live in chunks that are ambiguous out of context.

The corpus is 12 synthetic quarterly reports (3 companies × 4 quarters). Inside
each report, the section bodies deliberately never repeat the company name or
quarter — exactly like real filings, where a chunk says "Revenue grew 12%"
without saying who or when. A question like "How much did Acme's revenue grow
in Q3 2025?" is therefore unanswerable for plain chunk retrieval beyond luck,
while the situating blurb ("This chunk is from Acme Corp's Q3 2025 report…")
restores the missing signal for both embeddings and BM25.

Run (needs an endpoint — see .env.example; costs well under $1):

    python examples/benchmark.py             # full run, incl. LLM rerank
    python examples/benchmark.py --no-rerank # skip the rerank condition
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from contextual_rag import (
    ContextualChunk,
    VectorStore,
    answer_from,
    chunk_markdown,
    contextualize_chunks,
    llm,
    search_trace,
)

COMPANIES = ["Acme Corp", "Borealis Industries", "Cetus Analytics"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
TOP_K = 5  # ranking depth evaluated; Pass@k is computed for k in {1, 3, 5}

# ----------------------------------------------------------------- corpus ----

_FILLER = (
    "Management attributes the quarter's trajectory to disciplined execution, "
    "steady demand across the installed base, and continued investment in the "
    "product roadmap. The figures discussed below are unaudited and reflect "
    "continuing operations only. Comparative amounts have been restated where "
    "required by recently adopted accounting standards, and percentages are "
    "computed on constant-currency figures unless stated otherwise. Additional "
    "detail is provided in the supplemental tables accompanying this report."
)

# Standard boilerplate, long enough that every document clears the provider's
# ~1024-token prompt-prefix caching threshold — so the benchmark also
# demonstrates the caching economics that make contextualization cheap.
_SAFE_HARBOR = (
    "This report contains forward-looking statements within the meaning of "
    "applicable securities laws, including statements regarding anticipated "
    "revenue growth, margin trajectory, hiring plans, product availability, and "
    "capital allocation. These statements are based on current expectations and "
    "assumptions that are subject to risks and uncertainties, and actual "
    "results could differ materially. Factors that could cause such differences "
    "include macroeconomic conditions, competitive dynamics, customer demand, "
    "supply availability, foreign-exchange movements, and the other risks "
    "described in the company's most recent annual filing. The company "
    "undertakes no obligation to update any forward-looking statement, whether "
    "as a result of new information, future events, or otherwise. Non-GAAP "
    "measures referenced in this report are reconciled to their nearest GAAP "
    "equivalents in the appendix."
)


def _metrics(ci: int, qi: int) -> dict:
    """Deterministic metrics per (company, quarter) — no randomness, so every
    run sees the exact same corpus. Every formula is injective over
    ci ∈ {0,1,2}, qi ∈ {0..3} (coefficients chosen so ranges can't overlap):
    no two documents ever share a metric value, hence no two section bodies
    are ever byte-identical."""
    return {
        "growth": 4 + 8 * ci + 2 * qi,          # revenue growth %
        "revenue": 420 + 140 * ci + 35 * qi,    # $M
        "margin": 18 + 4 * ci + qi,             # operating margin %
        "hires": 40 + 45 * ci + 10 * qi,        # net new hires
        "headcount": 2600 + 900 * ci + 60 * qi,
        "guidance": 3 + 4 * ci + qi,            # next-quarter growth guidance %
    }


def make_doc(company: str, quarter: str) -> tuple[str, str]:
    """One synthetic quarterly report. Only the title/overview names the
    company and quarter — every other section is deliberately context-free."""
    ci, qi = COMPANIES.index(company), QUARTERS.index(quarter)
    m = _metrics(ci, qi)
    doc_id = f"{company.split()[0].lower()}-{quarter.lower()}-2025.md"
    md = f"""# {company} — {quarter} 2025 Quarterly Report

## Business overview

{company} closed {quarter} 2025 in line with the plan communicated at the start
of the fiscal year. This report covers the consolidated results for the quarter,
segment performance, and the outlook for the coming period. {_FILLER}

## Forward-looking statements

{_SAFE_HARBOR}

## Revenue

Total revenue grew {m["growth"]}% year over year, reaching ${m["revenue"]} million
for the quarter. Subscription lines outpaced services, and net revenue retention
remained above target. Deferred revenue expanded in step with multi-year
commitments signed late in the quarter. {_FILLER}

## Operating margin

Operating margin came in at {m["margin"]}% for the quarter, reflecting the cost
program initiated last year and a favorable services mix. Gross margin was
stable sequentially, while operating expenses grew slower than revenue.
{_FILLER}

## Headcount

The workforce grew by {m["hires"]} net new roles during the quarter, ending at
{m["headcount"]} employees. Hiring concentrated in engineering and go-to-market;
attrition stayed below the industry benchmark. {_FILLER}

## Segment results

The platform segment again contributed the majority of consolidated revenue,
followed by services and licensing. Segment operating income improved on volume
leverage, partially offset by planned pricing actions in the entry tier.
{_FILLER}

## Regional performance

Growth was broad-based across regions, with the strongest contribution from
EMEA, followed by North America and APAC. Currency movements had an immaterial
effect on reported figures for the quarter. {_FILLER}

## Outlook

For the next quarter, management guides to revenue growth of {m["guidance"]}%
year over year at the midpoint, with operating margin expected to remain within
one point of the current level. Guidance assumes no material change in the
demand environment. {_FILLER}

## Risk factors

Results may differ materially due to macroeconomic conditions, competitive
dynamics, foreign exchange, and the timing of large transactions. Investors
should read this section together with the annual report's full risk
disclosures. {_FILLER}
"""
    return doc_id, md


# ---------------------------------------------------------------- queries ----

# (question template, gold-section keyword)
TOPICS = [
    ("How much did {company}'s revenue grow in {quarter} 2025?", "Revenue"),
    ("What operating margin did {company} report in {quarter} 2025?", "Operating margin"),
    ("How did {company}'s headcount change in {quarter} 2025?", "Headcount"),
]


def build_queries() -> list[tuple[str, str, str]]:
    """All 36 (question, gold_doc_id, gold_section_keyword) triples."""
    out = []
    for company in COMPANIES:
        for quarter in QUARTERS:
            doc_id = f"{company.split()[0].lower()}-{quarter.lower()}-2025.md"
            for template, keyword in TOPICS:
                out.append((template.format(company=company, quarter=quarter),
                            doc_id, keyword))
    return out


def gold_ids(chunks_by_doc: dict, doc_id: str, keyword: str) -> set[str]:
    """Gold = chunks of the gold document whose section matches the topic."""
    return {c.chunk_id for c in chunks_by_doc[doc_id] if keyword in c.section}


# ------------------------------------------------------------- evaluation ----

def pass_at(ranked_ids: list[str], gold: set[str], k: int) -> bool:
    return any(cid in gold for cid in ranked_ids[:k])


def rank_dense(store: VectorStore, query: str) -> list[str]:
    return [c.chunk_id for c in store.query(llm.embed([query])[0], TOP_K)]


def rank_hybrid(store: VectorStore, query: str, rerank_mode: str) -> list[str]:
    trace = search_trace(query, k=TOP_K, store=store, rerank_mode=rerank_mode)
    return [c.chunk_id for c in trace.final]


def evaluate(name: str, queries, chunks_by_doc, rank_fn) -> dict:
    hits = {k: 0 for k in (1, 3, 5)}
    for query, doc_id, keyword in queries:
        ranked = rank_fn(query)
        gold = gold_ids(chunks_by_doc, doc_id, keyword)
        for k in hits:
            hits[k] += pass_at(ranked, gold, k)
    n = len(queries)
    print(f"  {name:34s} " + "  ".join(f"P@{k} {100 * hits[k] / n:5.1f}%" for k in hits))
    return {k: hits[k] / n for k in hits}


# ------------------------------------------------------------------- main ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the (slower) LLM-rerank condition")
    args = ap.parse_args()

    docs = [make_doc(c, q) for c in COMPANIES for q in QUARTERS]
    chunks_by_doc = {doc_id: chunk_markdown(md, doc_id=doc_id) for doc_id, md in docs}
    n_chunks = sum(len(v) for v in chunks_by_doc.values())
    print(f"corpus: {len(docs)} documents, {n_chunks} chunks\n")

    root = tempfile.mkdtemp(prefix="crag-bench-")

    # Plain store: identical chunks, no blurbs, no LLM involved.
    plain = VectorStore("plain", path=root)
    for doc_id, _ in docs:
        plain.add([ContextualChunk(chunk=c, context="") for c in chunks_by_doc[doc_id]])

    # Contextual store: the actual pipeline (blurb per chunk, prefix-cached).
    # concurrency=1 (sequential) maximizes cache hits on SHORT documents like
    # these: a provider's cache entry takes a few seconds to become readable
    # after the priming call, and parallel workers on a 10-chunk document
    # outrun it. On real multi-page documents the default concurrency wins.
    contextual = VectorStore("contextual", path=root)
    spent = cached = 0
    for doc_id, md in docs:
        ctx = contextualize_chunks(md, chunks_by_doc[doc_id], concurrency=1)
        spent += sum(c.total_tokens for c in ctx)
        cached += sum(c.cached_tokens for c in ctx)
        contextual.add(ctx)
    print(f"contextualization: {spent:,} tokens total, "
          f"{cached:,} served from the prompt-prefix cache "
          f"({100 * cached / max(1, spent):.0f}%)\n")

    queries = build_queries()
    print(f"evaluating {len(queries)} questions, Pass@k over top-{TOP_K}:\n")
    results = {}
    results["plain dense"] = evaluate(
        "plain — dense only", queries, chunks_by_doc, lambda q: rank_dense(plain, q))
    results["plain hybrid"] = evaluate(
        "plain — hybrid (dense+BM25+RRF)", queries, chunks_by_doc,
        lambda q: rank_hybrid(plain, q, "off"))
    results["ctx dense"] = evaluate(
        "contextual — dense only", queries, chunks_by_doc, lambda q: rank_dense(contextual, q))
    results["ctx hybrid"] = evaluate(
        "contextual — hybrid", queries, chunks_by_doc,
        lambda q: rank_hybrid(contextual, q, "off"))
    if not args.no_rerank:
        results["ctx hybrid+rr"] = evaluate(
            "contextual — hybrid + LLM rerank", queries, chunks_by_doc,
            lambda q: rank_hybrid(contextual, q, "llm"))

    base, best = results["plain hybrid"][5], results.get("ctx hybrid+rr", results["ctx hybrid"])[5]
    if base < 1:
        print(f"\nretrieval failures (@5): {100 * (1 - base):.1f}% plain-hybrid → "
              f"{100 * (1 - best):.1f}% best contextual "
              f"({100 * ((1 - base) - (1 - best)) / (1 - base):.0f}% reduction)")

    # ------------------------------------------------- qualitative example ----
    question = "How much did Borealis Industries' revenue grow in Q2 2025?"
    print(f"\nqualitative example — {question!r}")
    for label, store in (("plain", plain), ("contextual", contextual)):
        top = search_trace(question, k=3, store=store, rerank_mode="off").final
        ans = answer_from(question, top)
        print(f"\n[{label}] top-1: {top[0].citation}  (section: {top[0].section})")
        if top[0].context:
            print(f"[{label}] blurb: {top[0].context}")
        print(f"[{label}] answer: {ans.text}")


if __name__ == "__main__":
    sys.exit(main())
