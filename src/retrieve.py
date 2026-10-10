"""Phase 4: find the passages that best answer a question.

Hybrid search:
- meaning-based: the question's embedding is sent to the Qdrant vector
  database, which returns the chunks with the most similar embeddings
  (cosine similarity), so "how to live long" can find a passage about
  longevity that shares no words with the question;
- keyword-based: BM25 over the same chunks, so exact Sanskrit terms
  ("Rasayana", "pitta") are found even when the embedding model does not
  understand them;
- the two rankings are merged with Reciprocal Rank Fusion: each chunk scores
  1/(RRF_K + rank) in each list it appears in. Ranks, not raw scores, are
  combined, because the two scores are on different scales.

Results keep the chunk's kind and attribution, so a translator's note is
never shown as Charaka's words.

Run:  python -m src.retrieve "What urges should not be suppressed?"
"""
from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src import config
from src.bm25 import BM25, tokenize
from src.chunk import Chunk, embedding_text, load_chunks
from src.index import COLLECTION, open_index
from src.structure import lesson_label

RRF_K = 60          # standard constant from the RRF paper; damps the weight of top ranks
CANDIDATES = 50     # how deep each ranking goes before fusion
METHODS = ("hybrid", "dense", "bm25")


@dataclass
class Result:
    chunk: Chunk
    score: float
    dense_rank: int | None    # 1-based place in meaning-based search, None if not in its top CANDIDATES
    bm25_rank: int | None     # same for keyword search


def citation(chunk: Chunk) -> str:
    """E.g. "Sutra Sthana, Lesson VII (Navegandharaniya), pp. 85-86"."""
    label = lesson_label(chunk.sthana, chunk.lesson)
    printed = chunk.printed_pages
    numbers = [int(p) for p in printed if p.isdigit()]
    # Trust printed numbers only when every page has one and they run on
    # without gaps; OCR sometimes misreads them (e.g. "129" for "1015").
    trusted = (len(numbers) == len(printed) == len(chunk.scan_pages)
               and all(b == a + 1 for a, b in zip(numbers, numbers[1:])))
    if trusted:
        pages = f"p. {numbers[0]}" if len(numbers) == 1 else f"pp. {numbers[0]}-{numbers[-1]}"
    else:   # printed numbers missing or doubtful: fall back to the scan
        first, last = chunk.scan_pages[0], chunk.scan_pages[-1]
        pages = f"scan p. {first}" if first == last else f"scan pp. {first}-{last}"
    return f"{label}, {pages}"


def _top_keyword(scores: np.ndarray, allowed: np.ndarray, n: int) -> list[int]:
    """Indices of the n best positive BM25 scores among allowed chunks."""
    scores = np.where(allowed & (scores > 0), scores, -np.inf)
    order = np.argsort(-scores, kind="stable")[:n]
    return [int(i) for i in order if np.isfinite(scores[i])]


class Retriever:
    def __init__(self, chunks: list[Chunk], store: QdrantClient, embedder):
        """chunks[i] must be the chunk stored as Qdrant point i (see src/index.py)."""
        self.chunks = chunks
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25([tokenize(embedding_text(c)) for c in chunks])
        self._kinds = np.array([c.kind for c in chunks])

    @classmethod
    def load(cls, chunks_path: Path = config.CHUNKS_PATH, index_dir: Path = config.INDEX_DIR,
             url: str | None = config.QDRANT_URL, embedder=None) -> "Retriever":
        """The retriever the programs use: the Qdrant server if QDRANT_URL is set."""
        if embedder is None:
            from src.embedder import Embedder
            embedder = Embedder()
        chunks = load_chunks(chunks_path)
        return cls(chunks, open_index(chunks, embedder.model_name, index_dir, url), embedder)

    def close(self) -> None:
        """Release the database connection (an embedded database allows one program at a time)."""
        self.store.close()

    def search(self, question: str, k: int = 5, method: str = "hybrid",
               kinds: tuple[str, ...] = ("text", "note")) -> list[Result]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        allowed = np.isin(self._kinds, kinds)

        dense: list[int] = []
        if method in ("hybrid", "dense"):
            hits = self.store.query_points(
                COLLECTION, query=self.embedder.embed_query(question).tolist(), limit=CANDIDATES,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds)))]),
                with_payload=False,
            ).points
            dense = [int(hit.id) for hit in hits]
        keyword: list[int] = []
        if method in ("hybrid", "bm25"):
            keyword = _top_keyword(self.bm25.scores(tokenize(question)), allowed, CANDIDATES)

        dense_rank = {i: r for r, i in enumerate(dense, 1)}
        bm25_rank = {i: r for r, i in enumerate(keyword, 1)}
        fused: dict[int, float] = {}
        for ranks in (dense_rank, bm25_rank):
            for i, rank in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
        best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [Result(self.chunks[i], fused[i], dense_rank.get(i), bm25_rank.get(i)) for i in best]


class SerialSearch:
    """Lets threads (API requests, parallel evaluation) share one retriever.

    Searches run one at a time: they take milliseconds, and the embedded
    database and the embedder are not meant to be shared between threads. The
    slow part, the answer model, runs outside and in parallel.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._lock = threading.Lock()

    def search(self, *args, **kwargs) -> list[Result]:
        with self._lock:
            return self.retriever.search(*args, **kwargs)


def format_result(number: int, result: Result, preview: int = 300) -> str:
    chunk = result.chunk
    tag = "CHARAKA" if chunk.kind == "text" else "TRANSLATOR'S NOTE - not Charaka's words"
    ranks = ", ".join(f"{name} #{rank}" for name, rank in
                      (("meaning", result.dense_rank), ("keyword", result.bm25_rank)) if rank)
    text = " ".join(chunk.text.split())
    text = text if len(text) <= preview else text[:preview].rsplit(" ", 1)[0] + " ..."
    return (f"{number}. [{tag}] {citation(chunk)}\n"
            f"   {chunk.attribution} | {chunk.chunk_id} | {ranks}\n"
            f"   {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the Charaka Samhita.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of passages (default 5)")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--text-only", action="store_true", help="leave out translator's notes")
    args = parser.parse_args(argv)

    retriever = Retriever.load()
    kinds = ("text",) if args.text_only else ("text", "note")
    try:
        for number, result in enumerate(retriever.search(args.question, args.k, args.method, kinds), 1):
            print(format_result(number, result) + "\n")
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
