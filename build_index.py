"""
One-shot indexer: chunk the corpus, embed it, and persist it to ChromaDB.

Run once before querying (and again whenever you change documents/):
    python build_index.py
"""
from src.store import build_index

if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into ChromaDB (./chroma_db). Ready to query.")
