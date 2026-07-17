"""Okapi BM25 — the lexical (keyword) half of hybrid search.

Pure Python, no model, no GPU, no extra dependency. Builds an in-memory inverted index
over the chunks' contextual text and scores by term frequency × inverse document
frequency with length normalization. Fast and tiny at corpus scale (thousands of chunks).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tokens: list[list[str]] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0

    def build(self, records: list[tuple[str, str]]) -> "BM25":
        """records: list of (chunk_id, text)."""
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
