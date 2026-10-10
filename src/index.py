"""Phase 4: embed every chunk once and store it in the Qdrant vector database.

Input : data/processed/chunks.jsonl   (Phase 3)
Output: collection "charaka_chunks" in the Qdrant server at QDRANT_URL (Phase 6,
        see docker-compose.yml), or, when QDRANT_URL is not set, in the
        embedded database storage/index/qdrant/

Each chunk becomes one Qdrant "point":
  id      = the chunk's position in chunks.jsonl
  vector  = its 384-number embedding, compared by cosine similarity
  payload = the chunk's fields (lesson, pages, kind, text, ...), so the
            database can filter: kind = "text" leaves out translator's notes.

The collection's metadata records the embedding model and a fingerprint of
the chunks, so the retriever refuses an index built from different chunks or
another model: re-chunking without re-indexing fails loudly instead of
returning the wrong passages.

Two ways to run Qdrant, same code:
- server (QDRANT_URL set): a separate program, here in Docker; any number of
  programs (API server, chat, evaluation scripts) can use it at once;
- embedded (no QDRANT_URL): a folder opened by the Python library itself; no
  server needed, but only one program at a time. Tests use this mode.

Run:  python -m src.index                   (embed all chunks: 5-10 minutes on a laptop CPU)
      python -m src.index --from-embedded   (copy an existing embedded index to the
                                             server without embedding again)
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


class IndexNotReadyError(RuntimeError):
    """The search index cannot be used (missing, out of date, or unreachable)."""


class StaleIndexError(IndexNotReadyError):
    """The stored index is missing or does not match the current chunks or model."""


class QdrantUnavailableError(IndexNotReadyError):
    """The Qdrant server at QDRANT_URL does not answer."""


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\n{embedding_text(chunk)}\n".encode("utf-8"))
    return digest.hexdigest()


def open_store(index_dir: Path = config.INDEX_DIR, url: str | None = None) -> QdrantClient:
    """The Qdrant server at url, or else the embedded database in index_dir.

    Library functions default to the embedded database; the programs (index,
    chat, API server, ...) pass config.QDRANT_URL.
    """
    if url:
        return QdrantClient(url=url, api_key=config.QDRANT_API_KEY, timeout=30)
    return QdrantClient(path=str(index_dir / "qdrant"))


def _where(index_dir: Path, url: str | None) -> str:
    return f"Qdrant server {url}" if url else f"embedded database {index_dir / 'qdrant'}"


def _check_reachable(store: QdrantClient, url: str | None) -> None:
    if not url:
        return
    try:
        store.get_collections()
    except Exception as error:
        store.close()
        raise QdrantUnavailableError(
            f"Cannot reach the Qdrant server at {url}. Start it with: docker compose up -d qdrant"
        ) from error


def save_index(vectors: np.ndarray, chunks: list[Chunk], model_name: str,
               index_dir: Path = config.INDEX_DIR, url: str | None = None) -> None:
    """Replace the collection with these chunks and their vectors."""
    store = open_store(index_dir, url)
    _check_reachable(store, url)
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


def open_index(chunks: list[Chunk], model_name: str, index_dir: Path = config.INDEX_DIR,
               url: str | None = None) -> QdrantClient:
    """Open the database after checking it was built from these chunks with this model."""
    if not url and not (index_dir / "qdrant").exists():
        raise StaleIndexError(f"No search index in the {_where(index_dir, url)}. Run: python -m src.index")
    store = open_store(index_dir, url)
    _check_reachable(store, url)
    try:
        if not store.collection_exists(COLLECTION):
            raise StaleIndexError(f"No search index in the {_where(index_dir, url)}. "
                                  "Run: python -m src.index")
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


def read_vectors(store: QdrantClient, count: int) -> np.ndarray:
    """All stored vectors, in point-id (= chunk) order."""
    points, _ = store.scroll(COLLECTION, limit=count, with_vectors=True, with_payload=False)
    points.sort(key=lambda p: int(p.id))
    return np.array([p.vector for p in points], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed all chunks into the vector database.")
    parser.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR,
                        help="embedded database folder (used when QDRANT_URL is not set)")
    parser.add_argument("--from-embedded", action="store_true",
                        help="copy the embedded index to the Qdrant server instead of embedding again")
    args = parser.parse_args(argv)
    url = config.QDRANT_URL

    if not args.chunks.exists():
        print(f"Chunks not found: {args.chunks}")
        print("Run first: python -m src.chunk")
        return 1
    chunks = load_chunks(args.chunks)

    from src.embedder import MODEL_NAME, Embedder
    try:
        if args.from_embedded:
            if not url:
                print("Set QDRANT_URL in .env to the server to copy to.")
                return 1
            embedded = open_index(chunks, MODEL_NAME, args.index_dir)   # checks it is up to date
            try:
                vectors = read_vectors(embedded, len(chunks))
            finally:
                embedded.close()
            model_name = MODEL_NAME
        else:
            embedder = Embedder()
            texts = [embedding_text(c) for c in chunks]
            started = time.time()
            parts = []
            for start in range(0, len(texts), BATCH):
                parts.append(embedder.embed_passages(texts[start:start + BATCH]))
                done = min(start + BATCH, len(texts))
                print(f"\rEmbedded {done}/{len(texts)} chunks ({time.time() - started:.0f}s)",
                      end="", flush=True)
            print()
            vectors = np.vstack(parts)
            model_name = embedder.model_name
        save_index(vectors, chunks, model_name, args.index_dir, url)
    except IndexNotReadyError as error:
        print(error)
        return 1
    print(f"Stored {len(chunks)} chunks ({vectors.shape[1]}-number vectors) in collection "
          f"'{COLLECTION}' of the {_where(args.index_dir, url)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
