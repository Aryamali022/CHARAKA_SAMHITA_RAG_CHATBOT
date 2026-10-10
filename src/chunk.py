"""Phase 3: cut each lesson into passages ("chunks") for search and citation.

Input : data/processed/lesson_pages.jsonl  (Phase 2)
Output: data/processed/chunks.jsonl        (one JSON object per chunk)

Method: structure-aware paragraph packing.
- A chunk never crosses a lesson, so every chunk has one correct citation.
- The pages of a lesson are read as one continuous text: a paragraph cut by
  a page break (no full stop at the end of the page) is joined back together,
  and a word hyphenated across the break ("ac-" + "cording") is rejoined.
- Whole paragraphs are packed into chunks that fit the embedding model:
  context line + passage <= MAX_TOKENS tokens of its tokenizer. Only a single
  paragraph longer than that is split, at sentence ends where possible.
  (Sizes are measured in tokens, not words: Sanskrit words with diacritics
  split into many tokens, so a word limit cannot guarantee the fit.)
- The last paragraph of a chunk is repeated at the start of the next one when
  it is short, so an idea on a boundary appears whole at least once.
- The translator's footnotes are chunked separately, as kind "note", with an
  attribution saying they are not part of the Charaka Samhita. Answers that
  use them must say so.

Run:  python -m src.chunk
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from src import config
from src.structure import STHANAS, LessonPage, lesson_label, load_lesson_pages

# Sizes are in units of a `measure` function: tokens of the embedding model in
# the real pipeline (see src/embedder.py), words in the unit tests.
Measure = Callable[[str], int]
MAX_TOKENS = 512      # embedding model input limit, for context line + passage
SPECIAL_TOKENS = 2    # [CLS] and [SEP], added by the model's tokenizer
OVERLAP_TOKENS = 100  # longest paragraph repeated at the start of the next chunk

KINDS = {
    # kind: (field of LessonPage, attribution shown with every answer that uses it)
    "text": ("main_text",
             "Charaka Samhita, English translation by A. C. Kaviratna"),
    "note": ("notes_text",
             "Translator's note by A. C. Kaviratna, not part of the Charaka Samhita"),
}

# A page ends a sentence when its last line ends with . ? or ! (perhaps followed
# by a closing quote or bracket and a verse number such as "40" or "22-23").
SENTENCE_END_RE = re.compile(r"[.?!][\"'”’)\]]*(\s+\d+(\s*[-–]\s*\d+)?\.?)?$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[\"“‘(A-Z])")
HYPHEN_BREAK_RE = re.compile(r"\w-$")


@dataclass
class Chunk:
    chunk_id: str               # e.g. "sutra-01-text-003"
    source_id: str
    kind: str                   # "text" (Charaka) or "note" (translator)
    attribution: str            # who wrote this text; show it with the answer
    sthana: str                 # Sthana.key
    lesson: int
    lesson_title: str
    scan_pages: list[int]       # pages the text comes from
    printed_pages: list[str]    # the numbers printed on those pages, where known
    context: str                # one line naming the lesson; prepend it for embedding
    text: str                   # the book's own words
    word_count: int
    size: int                   # embedding_text() as measured when chunking (tokens)


@dataclass
class Paragraph:
    text: str
    pages: list[int]


def word_count(text: str) -> int:
    return len(text.split())


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded: the context line, then the passage."""
    return f"{chunk.context}\n\n{chunk.text}"


def lesson_paragraphs(records: list[LessonPage], field: str) -> list[Paragraph]:
    """The lesson's paragraphs in reading order, joined across page breaks."""
    paragraphs: list[Paragraph] = []
    continues = False
    for record in records:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", getattr(record, field)) if b.strip()]
        for i, block in enumerate(blocks):
            if i == 0 and continues:
                previous = paragraphs[-1]
                if HYPHEN_BREAK_RE.search(previous.text) and block[0].islower():
                    text = previous.text[:-1] + block
                else:
                    text = f"{previous.text} {block}"
                paragraphs[-1] = Paragraph(text, previous.pages + [record.scan_page])
            else:
                paragraphs.append(Paragraph(block, [record.scan_page]))
        continues = bool(blocks) and not SENTENCE_END_RE.search(blocks[-1])
    return paragraphs


def _split_words(text: str, limit: int, measure: Measure) -> list[str]:
    pieces: list[list[str]] = [[]]
    size = 0
    for word in text.split():
        word_size = measure(word)
        if pieces[-1] and size + word_size > limit:
            pieces.append([])
            size = 0
        pieces[-1].append(word)
        size += word_size
    return [" ".join(words) for words in pieces]


def split_long_paragraph(paragraph: Paragraph, limit: int, measure: Measure) -> list[Paragraph]:
    """Split a paragraph larger than limit at sentence ends (words as a last resort).

    Sizes add up across whitespace (true for word counts and for the BERT-style
    tokenizer of the embedding model), so pieces are sized by summing.
    """
    if measure(paragraph.text) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in SENTENCE_SPLIT_RE.split(paragraph.text):
        for part in _split_words(sentence, limit, measure):
            part_size = measure(part)
            if current and size + part_size > limit:
                pieces.append(" ".join(current))
                current, size = [], 0
            current.append(part)
            size += part_size
    pieces.append(" ".join(current))
    return [Paragraph(piece, paragraph.pages) for piece in pieces]


def pack(paragraphs: list[Paragraph], limit: int, measure: Measure,
         overlap: int = OVERLAP_TOKENS) -> list[list[Paragraph]]:
    """Group paragraphs into chunks of at most limit (each paragraph must fit on its own)."""
    sizes = {id(p): measure(p.text) for p in paragraphs}
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    for paragraph in paragraphs:
        size = sizes[id(paragraph)]
        if current and sum(sizes[id(p)] for p in current) + size > limit:
            groups.append(current)
            last = current[-1]
            keep = sizes[id(last)] <= overlap and sizes[id(last)] + size <= limit
            current = [last] if keep else []
        current.append(paragraph)
    if current:
        groups.append(current)
    return groups


def chunk_lesson(records: list[LessonPage], kind: str, measure: Measure,
                 max_size: int = MAX_TOKENS) -> list[Chunk]:
    field, attribution = KINDS[kind]
    first = records[0]
    label = lesson_label(first.sthana, first.lesson)
    context = f"Charaka Samhita, {label}" + (" - translator's note" if kind == "note" else "")
    limit = max_size - SPECIAL_TOKENS - measure(context)   # room left for the passage
    paragraphs = [piece for p in lesson_paragraphs(records, field)
                  for piece in split_long_paragraph(p, limit, measure)]
    printed = {r.scan_page: r.printed_page for r in records}

    chunks = []
    for number, group in enumerate(pack(paragraphs, limit, measure), 1):
        pages = sorted({page for p in group for page in p.pages})
        text = "\n\n".join(p.text for p in group)
        chunks.append(Chunk(
            chunk_id=f"{first.sthana}-{first.lesson:02d}-{kind}-{number:03d}",
            source_id=first.source_id, kind=kind, attribution=attribution,
            sthana=first.sthana, lesson=first.lesson, lesson_title=first.lesson_title,
            scan_pages=pages, printed_pages=[printed[p] for p in pages if printed[p]],
            context=context, text=text, word_count=word_count(text),
            size=measure(f"{context}\n\n{text}") + SPECIAL_TOKENS,
        ))
    return chunks


def build_chunks(records: list[LessonPage], measure: Measure,
                 max_size: int = MAX_TOKENS) -> list[Chunk]:
    by_lesson: dict[tuple[str, int], list[LessonPage]] = {}
    for record in records:
        by_lesson.setdefault((record.sthana, record.lesson), []).append(record)
    chunks = []
    for lesson_records in by_lesson.values():
        lesson_records.sort(key=lambda r: r.scan_page)
        for kind in KINDS:
            chunks.extend(chunk_lesson(lesson_records, kind, measure, max_size))
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def print_report(chunks: list[Chunk]) -> None:
    for kind in KINDS:
        own = [c for c in chunks if c.kind == kind]
        words = [c.word_count for c in own]
        tokens = [c.size for c in own]
        print(f"{kind:<4} chunks: {len(own):>5}  words median {statistics.median(words):.0f} "
              f"(max {max(words)}), tokens median {statistics.median(tokens):.0f} "
              f"(max {max(tokens)} of {MAX_TOKENS})")
    per_sthana = Counter((c.sthana, c.kind) for c in chunks)
    for sthana in STHANAS:
        print(f"  {sthana.name:<16}: {per_sthana[(sthana.key, 'text')]:>4} text, "
              f"{per_sthana[(sthana.key, 'note')]:>4} note")
    multi_page = sum(len(c.scan_pages) > 1 for c in chunks)
    print(f"Chunks spanning more than one page: {multi_page}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut lessons into chunks.")
    parser.add_argument("--lesson-pages", type=Path, default=config.LESSON_PAGES_PATH)
    parser.add_argument("--out", type=Path, default=config.CHUNKS_PATH)
    args = parser.parse_args(argv)

    if not args.lesson_pages.exists():
        print(f"Lesson pages not found: {args.lesson_pages}")
        print("Run first: python -m src.structure")
        return 1

    from src.embedder import Embedder   # loads the model (downloads it on first use)
    chunks = build_chunks(load_lesson_pages(args.lesson_pages), Embedder().count_tokens)
    write_chunks(chunks, args.out)
    print_report(chunks)
    print(f"\nWrote {len(chunks)} chunks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
