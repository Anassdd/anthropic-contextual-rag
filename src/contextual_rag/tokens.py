"""Token counting — exact via tiktoken when available, heuristic otherwise.

Kept pluggable (the ``chunk_*`` functions accept a ``count_tokens`` callable)
so the chunker never hard-depends on tiktoken's downloadable vocabulary — a
locked-down machine may not be able to fetch it. This module is just the
default counter; install the ``tokens`` extra for exact counts.
"""

from __future__ import annotations

_encoder = None
_tried = False


def _get_encoder():
    """Load the tiktoken encoder once; remember failure so we never retry."""
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
    """Count tokens in ``text``.

    Exact if tiktoken loads; otherwise the classic ~4-chars-per-token estimate.
    The estimate is slightly conservative on purpose — it keeps chunks at or
    under their size target rather than over.

    Args:
        text: The text to measure. Empty or None-ish counts as 0.

    Returns:
        The token count (>= 1 for any non-empty text).
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / 4))
