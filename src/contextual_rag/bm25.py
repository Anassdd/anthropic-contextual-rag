"""Okapi BM25 — the lexical (keyword) half of hybrid search.

Pure Python: no model, no GPU, no extra dependency, no search server. Builds an
in-memory inverted index over the chunks' contextual text and scores documents
by term frequency × inverse document frequency with length normalization
(the classic Robertson–Spärck Jones formulation).

Why it earns its place next to embeddings: BM25 catches *exact* terms — error
codes, function names, ticker symbols, accented French vocabulary — that dense
embeddings blur into their neighborhood. Fused with dense retrieval via RRF,
each side covers the other's misses. At corpus scale (thousands of chunks) a
full scoring pass is well under a millisecond-per-query regime; no index
server is worth its operational cost here.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens; ``\\w`` keeps Unicode letters (accents survive)."""
    return _TOKEN.findall((text or "").lower())


class BM25:
    """An in-memory Okapi BM25 index over ``(chunk_id, text)`` records.

    Args:
        k1: Term-frequency saturation. Higher = repeated terms keep counting.
        b: Length normalization strength. 1 = full penalty for long documents,
            0 = none. The defaults (1.5, 0.75) are the standard literature values.

    Example:
        >>> bm = BM25().build([("c1", "hybrid retrieval"), ("c2", "dense only")])
        >>> bm.search("hybrid", k=1)
        [('c1', ...)]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tokens: list[list[str]] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0

    def build(self, records: list[tuple[str, str]]) -> BM25:
        """Index ``records`` (a list of ``(chunk_id, text)``), replacing any
        previous index. Returns ``self`` for chaining."""
        self.ids = [cid for cid, _ in records]
        self.tokens = [_tokenize(t) for _, t in records]
        self.tf = [Counter(toks) for toks in self.tokens]
        self.doc_len = [len(toks) for toks in self.tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df = Counter()
        for toks in self.tokens:
            for term in set(toks):
                self.df[term] += 1
        n = len(records)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in self.df.items()}
        return self

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Score every indexed record against ``query``.

        Args:
            query: Free-text query; tokenized the same way as the records.
            k: Maximum number of results.

        Returns:
            Up to ``k`` ``(chunk_id, score)`` pairs, best first. Records with
            no query-term overlap are omitted entirely (score would be 0).
        """
        q = [t for t in _tokenize(query) if t in self.idf]
        if not q or not self.ids:
            return []
        scores = []
        for i, cid in enumerate(self.ids):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf[term] * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((cid, round(s, 4)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
