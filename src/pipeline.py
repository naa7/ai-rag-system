"""
End-to-end query pipeline: retrieve -> generate -> attribute.

`ask(question)` is the single entry point used by the Gradio UI (app.py) and the
evaluation script (evaluate.py).
"""
from __future__ import annotations

import os

from .config import ACTIVE_CONFIG
from .generate import generate_answer
from .store import retrieve

TOP_K = int(os.getenv("TOP_K", "5"))


def ask(question: str, k: int = TOP_K) -> dict:
    """
    Answer a question against the indexed corpus.

    Returns a dict with:
      - answer:  grounded answer text (or the "not enough information" sentence)
      - sources: de-duplicated list of source documents the answer can draw from,
                 in retrieval order (appended programmatically, not by the LLM)
      - chunks:  the raw retrieved chunks (text + metadata + distance) for inspection
    """
    chunks = retrieve(question, k=k)
    answer = generate_answer(question, chunks)

    # Programmatic source attribution: unique sources in retrieval order.
    seen, sources = set(), []
    for c in chunks:
        if c.label not in seen:
            seen.add(c.label)
            sources.append(c.label)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": [
            {
                "source": c.source,
                "label": c.label,
                "metadata": c.metadata,
                "distance": round(c.distance, 3),
                "text": c.text,
            }
            for c in chunks
        ],
    }
