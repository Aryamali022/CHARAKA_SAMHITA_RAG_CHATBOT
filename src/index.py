"""Phase 4: embed every chunk once and store it in the Qdrant vector database.

Input : data/processed/chunks.jsonl   (Phase 3)
Output: storage/index/qdrant/         (Qdrant database, collection "charaka_chunks")

Each chunk becomes one Qdrant "point":
  id      = the chunk's position in chunks.jsonl
  vector  = its 384-number embedding, compared by cosine similarity
  payload = the chunk's fields (lesson, pages, kind, text, ...), so the
            database can filter: kind = "text" leaves out translator's notes.

The collection's metadata records the embedding model and a fingerprint of
the chunks, so the retriever refuses an index built from different chunks or
another model: re-chunking without re-indexing fails loudly instead of
returning the wrong passages.

Qdrant runs embedded ("local mode"): no server to install, the database is a
folder. Pointing open_store() at a Qdrant server is the only change needed to
run it as a service.

Run:  python -m src.index        (about 5-10 minutes on a laptop CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src import config
from src.chunk import Chunk, embedding_text, load_chunks

COLLECTION = "charaka_chunks"
BATCH = 64


class StaleIndexError(RuntimeError):
    """The stored index does not match the current chunks or model."""


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\n{embedding_text(chunk)}\n".encode("utf-8"))
    return digest.hexdigest()


def open_store(index_dir: Path = config.INDEX_DIR) -> QdrantClient:
    """The Qdrant database in index_dir. Only one process can open it at a time."""
    return QdrantClient(path=str(index_dir / "qdrant"))


def save_index(vectors: np.ndarray, chunks: list[Chunk], model_name: str, index_dir: Path) -> None:
    """Replace the collection with these chunks and their vectors."""
    store = open_store(index_dir)
    try:
        if store.collection_exists(COLLECTION):
            store.delete_collection(COLLECTION)
        store.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
            metadata={"model": model_name, "chunks_sha256": chunks_fingerprint(chunks)},
        )
        store.upload_points(COLLECTION, (
            models.PointStruct(id=i, vector=vector.tolist(), payload=asdict(chunk))
            for i, (vector, chunk) in enumerate(zip(vectors, chunks))
        ), batch_size=BATCH, wait=True)
    finally:
        store.close()


def open_index(chunks: list[Chunk], model_name: str, index_dir: Path = config.INDEX_DIR) -> QdrantClient:
    """Open the database after checking it was built from these chunks with this model."""
    if not (index_dir / "qdrant").exists():
        raise StaleIndexError("No search index found. Run: python -m src.index")
    store = open_store(index_dir)
    try:
        if not store.collection_exists(COLLECTION):
            raise StaleIndexError("No search index found. Run: python -m src.index")
        metadata = store.get_collection(COLLECTION).config.metadata or {}
        if metadata.get("model") != model_name:
            raise StaleIndexError(f"Index was built with {metadata.get('model')}, not {model_name}. "
                                  "Run: python -m src.index")
        if (metadata.get("chunks_sha256") != chunks_fingerprint(chunks)
                or store.count(COLLECTION).count != len(chunks)):
            raise StaleIndexError("Chunks changed since the index was built. Run: python -m src.index")
    except Exception:
        store.close()
        raise
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed all chunks into the vector database.")
    parser.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    parser.add_argument("--out", type=Path, default=config.INDEX_DIR)
    args = parser.parse_args(argv)

    if not args.chunks.exists():
        print(f"Chunks not found: {args.chunks}")
        print("Run first: python -m src.chunk")
        return 1

    from src.embedder import Embedder
    chunks = load_chunks(args.chunks)
    embedder = Embedder()
    texts = [embedding_text(c) for c in chunks]
    started = time.time()
    parts = []
    for start in range(0, len(texts), BATCH):
        parts.append(embedder.embed_passages(texts[start:start + BATCH]))
        done = min(start + BATCH, len(texts))
        print(f"\rEmbedded {done}/{len(texts)} chunks ({time.time() - started:.0f}s)", end="", flush=True)
    vectors = np.vstack(parts)
    save_index(vectors, chunks, embedder.model_name, args.out)
    print(f"\nStored {len(chunks)} chunks ({vectors.shape[1]}-number vectors) in Qdrant "
          f"collection '{COLLECTION}' at {args.out / 'qdrant'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
