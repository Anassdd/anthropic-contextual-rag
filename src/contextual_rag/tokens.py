"""Token counting for the chunker — exact via tiktoken when available, else a heuristic.

Kept pluggable (chunk_* functions accept a `count_tokens` callable) so the chunker
never hard-depends on tiktoken's downloadable vocab — a locked-down machine may not
fetch it. This module is just the default counter.
"""

from __future__ import annotations

_encoder = None
_tried = False


def _get_encoder():
    global _encoder, _tried
    if _tried:
        return _encoder
    _tried = True
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("o200k_base")  # gpt-4o / gpt-5 family
    except Exception:
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Token count for `text`. Exact if tiktoken loads; otherwise the classic
    ~4-chars-per-token estimate (slightly conservative, which keeps chunks at/under
    target rather than over)."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / 4))
