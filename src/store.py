"""
Embedding + vector store + retrieval.

Pipeline stages 3 & 4:
  - build_index(): embed all chunks with all-MiniLM-L6-v2 and store them in a
                   persistent ChromaDB collection, with source metadata.
  - retrieve():    embed a query and return the top-k most similar chunks,
                   each with its text, metadata, and distance score.

The embedding model runs locally (no API key). ChromaDB persists to ./chroma_db.
The collection name comes from the active DomainConfig so different corpora stay
isolated. Chunk metadata is passed through as a free-form dict (domain-agnostic).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from .config import ACTIVE_CONFIG
from .ingest import build_chunks

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = ACTIVE_CONFIG.name
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

# Module-level singletons so the model loads only once per process.
_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float
    metadata: dict = field(default_factory=dict)  # all domain-specific header keys

    @property
    def label(self) -> str:
        """Human-readable source label per the active domain config."""
        return ACTIVE_CONFIG.label_for(self.metadata)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


def build_index(raw_dir: str | None = None) -> int:
    """
    (Re)build the vector index from the document corpus.

    Embeds every chunk and stores it in ChromaDB with metadata. Returns the
    number of chunks indexed. Safe to re-run: the collection is dropped first.
    Pass `raw_dir` to index a corpus other than the default documents/ folder.
    """
    chunks = build_chunks(raw_dir) if raw_dir else build_chunks()
    if not chunks:
        raise RuntimeError("No chunks produced — check documents/ contains .txt files.")

    client = get_client()
    # Start fresh so re-running never duplicates entries.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    model = get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    collection.add(
        ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def _get_collection() -> chromadb.Collection:
    try:
        return get_client().get_collection(COLLECTION_NAME)
    except Exception as exc:  # collection missing
        raise RuntimeError(
            "Vector index not found. Run `python build_index.py` first."
        ) from exc


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the top-k chunks most semantically similar to `query`."""
    collection = _get_collection()
    q_emb = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)

    out: list[RetrievedChunk] = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = dict(meta or {})
        out.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "?"),
                chunk_index=meta.get("chunk_index", -1),
                distance=float(dist),
                metadata=meta,
            )
        )
    return out


# Quick retrieval smoke test (Milestone 4 checkpoint).
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What do students say about Professor Smith's exams?"
    print(f"Query: {query}\n")
    for i, rc in enumerate(retrieve(query, k=5), 1):
        print(f"{i}. [{rc.label} | dist={rc.distance:.3f}]")
        print(f"   {rc.text[:200]}\n")
