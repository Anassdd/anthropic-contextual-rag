"""Structure-aware recursive Markdown chunker.

Cuts on real structure (Markdown headings, then paragraphs/sentences), size-bound to
~512 tokens, adds a small overlap so a sentence straddling a boundary survives, and
carries provenance (doc + page + section) on every chunk. Semantic/embedding chunking
is deliberately NOT used — slower, not consistently better; the cut is low-leverage,
the gains come later (contextual retrieval).

Atomic units are never split: fenced code, display math ($$…$$), and HTML tables stay
whole. LaTeX is kept inline as text. Output is `Chunk` objects ready to embed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextual_rag.tokens import count_tokens as _default_count
from contextual_rag.types import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HTML_TABLE_OPEN = re.compile(r"^\s*<table", re.I)
_HTML_TABLE_CLOSE = re.compile(r"</table>", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Block:
    kind: str  # "heading" | "content"
    text: str
    start: int
    end: int
    level: int = 0
    heading_text: str = ""
    atomic: bool = False  # code / $$math$$ / table — never split, even if over target


def _lines_with_offsets(text: str):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


def _blocks(markdown: str) -> list[_Block]:
    """Split Markdown into heading + content blocks, each tagged with its char span.
    Atomic constructs (code/math/table) are consumed whole."""
    lines = list(_lines_with_offsets(markdown))
    n = len(lines)
    blocks: list[_Block] = []
    i = 0
    while i < n:
        start, line = lines[i]
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue

        m = _HEADING.match(line.rstrip("\n"))
        if m:
            blocks.append(_Block("heading", stripped, start, start + len(line),
                                 level=len(m.group(1)), heading_text=m.group(2).strip()))
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            token = fence.group(1)
            buf, s, i = [line], start, i + 1
            while i < n:
                _, l2 = lines[i]
                buf.append(l2)
                i += 1
                if l2.strip().startswith(token):
                    break
            blocks.append(_Block("content", "".join(buf).strip("\n"), s,
                                 s + sum(len(x) for x in buf), atomic=True))
            continue

        if stripped.startswith("$$"):
            buf, s, i = [line], start, i + 1
            if stripped.count("$$") < 2:  # not closed on the same line
                while i < n:
                    _, l2 = lines[i]
                    buf.append(l2)
                    i += 1
                    if "$$" in l2:
                        break
            blocks.append(_Block("content", "".join(buf).strip("\n"), s,
                                 s + sum(len(x) for x in buf), atomic=True))
            continue

        if _HTML_TABLE_OPEN.match(line):
            buf, s, i = [line], start, i + 1
            while i < n and not _HTML_TABLE_CLOSE.search(buf[-1]):
                _, l2 = lines[i]
                buf.append(l2)
                i += 1
            blocks.append(_Block("content", "".join(buf).strip("\n"), s,
                                 s + sum(len(x) for x in buf), atomic=True))
            continue

        # Paragraph / list / Markdown table: run until a blank line or a new structure.
        buf, s, i = [line], start, i + 1
        while i < n:
            _, l2 = lines[i]
            t = l2.strip()
            if t == "" or _HEADING.match(l2.rstrip("\n")) or _FENCE.match(l2) \
                    or t.startswith("$$") or _HTML_TABLE_OPEN.match(l2):
                break
            buf.append(l2)
            i += 1
        blocks.append(_Block("content", "".join(buf).strip("\n"), s, s + sum(len(x) for x in buf)))
    return blocks


def _sections(blocks: list[_Block]):
    """Group content blocks under their heading path. Content before any heading gets []."""
    stack: list[tuple[int, str]] = []
    path: list[str] = []
    current: list[_Block] = []
    out: list[tuple[list[str], list[_Block]]] = []

    def flush():
        if current:
            out.append((list(path), list(current)))

    for b in blocks:
        if b.kind == "heading":
            flush()
            current.clear()
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.heading_text))
            path[:] = [h[1] for h in stack]
            # Keep the heading line in the body too, not only in the path: a chunk must
            # carry every character of the source so nothing is lost, searchable, or
            # un-citable — even if a line was mis-promoted to a heading upstream.
            current.append(_Block("content", b.text, b.start, b.end))
        else:
            current.append(b)
    flush()
    return out


def _hard_split(text: str, target: int, count) -> list[str]:
    chars = max(1, int(len(text) / max(1, count(text)) * target))
    return [text[i:i + chars] for i in range(0, len(text), chars)] or [text]


def _split_block(b: _Block, target: int, count) -> list[_Block]:
    """Split one oversized block into <=target pieces (sentences, then hard chars).
    Pieces inherit the parent span, so page provenance stays correct."""
    pieces: list[_Block] = []
    cur: list[str] = []
    cur_tok = 0

    def emit():
        if cur:
            pieces.append(_Block("content", " ".join(cur), b.start, b.end))

    for sent in (s for s in _SENTENCE.split(b.text) if s.strip()):
        stok = count(sent)
        if stok > target:
            emit()
            cur, cur_tok = [], 0
            for piece in _hard_split(sent, target, count):
                pieces.append(_Block("content", piece, b.start, b.end))
            continue
        if cur and cur_tok + stok > target:
            emit()
            cur, cur_tok = [], 0
        cur.append(sent)
        cur_tok += stok
    emit()
    return pieces


def _pack(blocks: list[_Block], target: int, min_tokens: int, count) -> list[list[_Block]]:
    groups: list[list[_Block]] = []
    cur: list[_Block] = []
    cur_tok = 0
    for b in blocks:
        btok = count(b.text)
        if btok > target:
            if cur:
                groups.append(cur)
                cur, cur_tok = [], 0
            if b.atomic:  # a formula / code block / table is emitted whole, never split
                groups.append([b])
            else:
                for piece in _split_block(b, target, count):
                    groups.append([piece])
            continue
        if cur and cur_tok + btok > target:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(b)
        cur_tok += btok
    if cur:
        groups.append(cur)

    if len(groups) >= 2:  # fold a too-small tail into the previous chunk
        tail = "\n\n".join(b.text for b in groups[-1])
        if count(tail) < min_tokens:
            groups[-2].extend(groups.pop())
    return groups


def _tail(text: str, overlap_tokens: int, count) -> str:
    """Trailing slice of `text` ~overlap_tokens long, snapped to sentence boundaries."""
    if overlap_tokens <= 0 or not text:
        return ""
    sentences = _SENTENCE.split(text)
    out: list[str] = []
    for s in reversed(sentences):
        out.insert(0, s)
        if count(" ".join(out)) >= overlap_tokens:
            break
    return " ".join(out).strip()


def _pages_for(spans, page_offsets) -> list[int]:
    if not page_offsets:
        return []
    pages = set()
    for bs, be in spans:
        for pno, ps, pe in page_offsets:
            if bs < pe and be > ps:
                pages.add(pno)
    return sorted(pages)


def chunk_markdown(
    markdown: str,
    *,
    doc_id: str = "document",
    domain_id: str = "default",
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 64,
    count_tokens=_default_count,
    page_offsets=None,
) -> list[Chunk]:
    """Chunk a Markdown string into provenance-tagged `Chunk`s.

    `page_offsets` (list of (page_no, start, end) char spans) maps chunks back to pages;
    pass it for page provenance, omit it for raw text. `count_tokens` is injectable.
    """
    sections = _sections(_blocks(markdown or ""))
    chunks: list[Chunk] = []
    index = 0
    for path, content in sections:
        groups = _pack(content, target_tokens, min_tokens, count_tokens)
        prev_body = None
        prev_atomic = False
        for gi, group in enumerate(groups):
            body = "\n\n".join(b.text for b in group)
            atomic = len(group) == 1 and group[0].atomic
            # Skip overlap across an atomic block: copying a formula/code/table fragment
            # forward would duplicate it and tear it apart.
            use_overlap = gi > 0 and prev_body and not prev_atomic and not atomic
            prefix = _tail(prev_body, overlap_tokens, count_tokens) if use_overlap else ""
            text = f"{prefix}\n\n{body}" if prefix else body
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{index}",
                doc_id=doc_id,
                index=index,
                text=text,
                header_path=list(path),
                pages=_pages_for([(b.start, b.end) for b in group], page_offsets),
                token_count=count_tokens(text),
                char_count=len(text),
                overlap_tokens=count_tokens(prefix) if prefix else 0,
                domain_id=domain_id,
            ))
            index += 1
            prev_body = body
            prev_atomic = atomic
    return chunks


def chunk_parsed_doc(
    doc,
    *,
    domain_id: str = "default",
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 64,
    count_tokens=_default_count,
) -> list[Chunk]:
    """Chunk a parser `ParsedDoc`, deriving page provenance from its per-page Markdown."""
    sep = "\n\n"
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    pos = 0
    for page_no, page_md in enumerate(doc.page_markdown, start=1):
        page_md = page_md or ""
        offsets.append((page_no, pos, pos + len(page_md)))
        parts.append(page_md)
        pos += len(page_md) + len(sep)
    return chunk_markdown(
        sep.join(parts),
        doc_id=getattr(doc, "filename", "document"),
        domain_id=domain_id,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        count_tokens=count_tokens,
        page_offsets=offsets,
    )
