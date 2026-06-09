"""
Document ingestion + chunking.

Pipeline stages 1 & 2:
  - load_documents(): read raw .txt files, parse the metadata header, return clean text
  - clean_text():     normalize whitespace / strip separators (also handles scraped noise)
  - chunk_document(): paragraph-aware greedy packer (~600 chars, ~80 char overlap)

Document/chunk metadata is generic (a free-form dict) so the same code serves any
corpus; which header keys are kept is driven by the active DomainConfig.

Run directly to inspect the corpus:
    python -m src.ingest          # or: python src/ingest.py
prints total chunk count + 5 sample chunks.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .config import ACTIVE_CONFIG

# Chunking parameters (see planning.md "Chunking Strategy" for the rationale).
# Primary unit = one review (paragraph). A review is a self-contained opinion, so
# each becomes its own chunk. Only reviews longer than MAX_CHARS are sentence-split,
# and those splits carry OVERLAP_CHARS of context across the boundary.
MAX_CHARS = 600      # split threshold: a single review longer than this is sentence-split
OVERLAP_CHARS = 80   # tail carried across a split boundary (snapped to a word boundary)
MIN_CHARS = 40       # discard anything shorter than this (fragments, stray lines)

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")


@dataclass
class Document:
    source: str            # filename, e.g. "rmp_smith.txt"
    text: str              # cleaned body (reviews only, header removed)
    metadata: dict = field(default_factory=dict)  # parsed header keys (domain-specific)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)  # includes source + chunk_index + header keys


# --------------------------------------------------------------------------- #
# Loading + cleaning
# --------------------------------------------------------------------------- #
def _parse_header(raw: str) -> tuple[dict, str]:
    """Split the 'Key: value' header from the review body (separated by '---')."""
    meta: dict[str, str] = {}
    if "---" in raw:
        header, body = raw.split("---", 1)
    else:
        header, body = "", raw
    for line in header.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip().lower()] = val.strip()
    return meta, body


def clean_text(text: str) -> str:
    """
    Normalize document text.

    The synthetic corpus is already clean, but this function is where real
    scraped noise would be removed, so it also strips common web artifacts:
    HTML tags, HTML entities, and collapsed whitespace.
    """
    text = re.sub(r"<[^>]+>", " ", text)                 # strip HTML tags
    text = (text.replace("&amp;", "&").replace("&nbsp;", " ")
                .replace("&#39;", "'").replace("&quot;", '"').replace("&gt;", ">")
                .replace("&lt;", "<"))                    # decode common entities
    text = re.sub(r"[ \t]+", " ", text)                  # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)               # collapse extra blank lines
    return text.strip()


def load_documents(raw_dir: str = RAW_DIR) -> list[Document]:
    """
    Load every .txt file in raw_dir into a Document with parsed metadata.

    The header is parsed generically as "Key: value". Which keys are retained is
    driven by the active DomainConfig (`metadata_fields`); an empty list keeps all
    of them. The `source` key is always present.
    """
    keep = set(ACTIVE_CONFIG.metadata_fields)
    docs: list[Document] = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as fh:
            raw = fh.read()
        header, body = _parse_header(raw)
        meta = {"source": fname}
        for key, val in header.items():
            if not keep or key in keep:
                meta[key] = val
        docs.append(Document(source=fname, text=clean_text(body), metadata=meta))
    return docs


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _word_boundary_tail(text: str, n: int) -> str:
    """Return roughly the last n chars of text, snapped to start at a word boundary."""
    tail = text[-n:]
    if " " in tail and len(text) > n:
        tail = tail[tail.index(" ") + 1:]   # drop the leading partial word
    return tail.strip()


def _split_long_paragraph(para: str) -> list[str]:
    """Split an over-long review on sentence boundaries, carrying word-aligned overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > MAX_CHARS:
            out.append(cur.strip())
            cur = _word_boundary_tail(cur, OVERLAP_CHARS) + " " + s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_document(doc: Document) -> list[Chunk]:
    """
    Review-based chunking.

    Reviews are separated by blank lines. Each review is a complete, self-contained
    opinion, so it becomes its own chunk — the natural semantic unit for this corpus.
    A review longer than MAX_CHARS is sentence-split, with ~OVERLAP_CHARS of
    word-aligned context carried across the split so a fact spanning the boundary
    stays retrievable.
    """
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    for p in paragraphs:
        chunks.extend(_split_long_paragraph(p) if len(p) > MAX_CHARS else [p])

    result: list[Chunk] = []
    idx = 0
    for c in chunks:
        if len(c) < MIN_CHARS:
            continue
        meta = {**doc.metadata, "chunk_index": idx}
        result.append(
            Chunk(text=c, source=doc.source, chunk_index=idx, metadata=meta)
        )
        idx += 1
    return result


def build_chunks(raw_dir: str = RAW_DIR) -> list[Chunk]:
    """Load every document and return the full list of chunks."""
    chunks: list[Chunk] = []
    for doc in load_documents(raw_dir):
        chunks.extend(chunk_document(doc))
    return chunks


# --------------------------------------------------------------------------- #
# Inspection CLI (Milestone 3 checkpoint)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random

    docs = load_documents()
    chunks = build_chunks()
    print(f"Loaded {len(docs)} documents from {RAW_DIR}")
    print(f"Produced {len(chunks)} chunks "
          f"(one review per chunk; split threshold {MAX_CHARS} chars, "
          f"overlap ~{OVERLAP_CHARS}).\n")

    lengths = [len(c.text) for c in chunks]
    if lengths:
        print(f"Chunk length: min={min(lengths)} max={max(lengths)} "
              f"avg={sum(lengths) // len(lengths)} chars\n")

    print("=== 5 sample chunks ===")
    for c in random.sample(chunks, min(5, len(chunks))):
        print(f"\n[{ACTIVE_CONFIG.label_for(c.metadata)} | chunk {c.chunk_index} "
              f"| {len(c.text)} chars]")
        print(c.text)
