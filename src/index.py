"""Phase 4: embed every chunk once and save the vectors.

Input : data/processed/chunks.jsonl   (Phase 3)
Output: storage/index/embeddings.npy  (one 384-number vector per chunk, in chunk order)
        storage/index/meta.json       (model name + a fingerprint of the chunks)

The fingerprint lets the retriever refuse an index built from different
chunks, so re-chunking without re-indexing fails loudly instead of returning
the wrong passages.

Run:  python -m src.index        (about 5-10 minutes on a laptop CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.chunk import Chunk, embedding_text, load_chunks

BATCH = 64


class StaleIndexError(RuntimeError):
    """The saved index does not match the current chunks or model."""


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\n{embedding_text(chunk)}\n".encode("utf-8"))
    return digest.hexdigest()


def save_index(vectors: np.ndarray, chunks: list[Chunk], model_name: str, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", vectors.astype(np.float32))
    meta = {"model": model_name, "chunks": len(chunks), "dimensions": int(vectors.shape[1]),
            "chunks_sha256": chunks_fingerprint(chunks)}
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_index(chunks: list[Chunk], model_name: str, index_dir: Path = config.INDEX_DIR) -> np.ndarray:
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        raise StaleIndexError("No search index found. Run: python -m src.index")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["model"] != model_name:
        raise StaleIndexError(f"Index was built with {meta['model']}, not {model_name}. "
                              "Run: python -m src.index")
    if meta["chunks_sha256"] != chunks_fingerprint(chunks):
        raise StaleIndexError("Chunks changed since the index was built. Run: python -m src.index")
    return np.load(index_dir / "embeddings.npy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed all chunks into the search index.")
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
    print(f"\nWrote {vectors.shape[0]} vectors of {vectors.shape[1]} numbers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
