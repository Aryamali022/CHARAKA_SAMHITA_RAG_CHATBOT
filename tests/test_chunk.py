import re

import pytest

from src import config
from src.chunk import (
    MAX_TOKENS,
    Paragraph,
    build_chunks,
    embedding_text,
    lesson_paragraphs,
    load_chunks,
    pack,
    split_long_paragraph,
    word_count,
    write_chunks,
)
from src.ingest import ingest_file
from src.structure import LessonPage, build_structure

# Unit tests measure size in words, so they do not need the embedding model.


def lesson_page(scan_page, text, notes="", lesson=1):
    return LessonPage(
        source_id="test", sthana="sutra", lesson=lesson, lesson_title=f"title {lesson}",
        scan_page=scan_page, printed_page=str(scan_page), main_text=text,
        notes_text=notes, flags=[],
    )


def sentence(n, word="word"):
    """A sentence of exactly n words."""
    return " ".join([word] * (n - 1) + ["end."])


def paragraphs(*sizes):
    return [Paragraph(sentence(size, f"p{i}"), [1]) for i, size in enumerate(sizes)]


def size(group):
    return sum(word_count(p.text) for p in group)


# --- reading a lesson as continuous text -----------------------------------

def test_paragraph_cut_by_a_page_break_is_joined():
    records = [lesson_page(1, "First paragraph.\n\nThe sentence continues"),
               lesson_page(2, "across the page.\n\nNext paragraph.")]
    result = lesson_paragraphs(records, "main_text")
    assert [p.text for p in result] == [
        "First paragraph.", "The sentence continues across the page.", "Next paragraph."]
    assert [p.pages for p in result] == [[1], [1, 2], [2]]


def test_word_hyphenated_across_pages_is_rejoined():
    records = [lesson_page(1, "as he had ac-"), lesson_page(2, "quired it.")]
    [paragraph] = lesson_paragraphs(records, "main_text")
    assert paragraph.text == "as he had acquired it."


@pytest.mark.parametrize("ending", ["ever.", "ever. 40", "ever. 22-23", "ever.”", "ever?"])
def test_page_ending_a_sentence_is_not_joined(ending):
    records = [lesson_page(1, f"It lasts for {ending}"), lesson_page(2, "New paragraph.")]
    assert len(lesson_paragraphs(records, "main_text")) == 2


# --- splitting and packing ------------------------------------------------

def test_long_paragraph_is_split_at_sentence_ends():
    paragraph = Paragraph(" ".join(sentence(10) for _ in range(50)), [1, 2])
    pieces = split_long_paragraph(paragraph, 100, word_count)
    assert all(word_count(p.text) <= 100 and p.text.endswith("end.") for p in pieces)
    assert sum(word_count(p.text) for p in pieces) == 500
    assert all(p.pages == [1, 2] for p in pieces)


def test_sentence_longer_than_the_limit_is_split_by_words():
    pieces = split_long_paragraph(Paragraph(sentence(250), [1]), 100, word_count)
    assert [word_count(p.text) for p in pieces] == [100, 100, 50]


def test_short_paragraph_is_never_split():
    assert split_long_paragraph(Paragraph(sentence(40), [1]), 100, word_count) == [
        Paragraph(sentence(40), [1])]


def test_split_respects_a_measure_that_is_not_word_count():
    """Sizes come from the measure function (in the pipeline: model tokens)."""
    double = lambda text: 2 * word_count(text)   # every word "costs" two tokens
    pieces = split_long_paragraph(Paragraph(sentence(100), [1]), 60, double)
    assert all(double(p.text) <= 60 for p in pieces)


def test_chunks_respect_the_limit_and_overlap_by_one_paragraph():
    groups = pack(paragraphs(*[50] * 20), 300, word_count, overlap=60)
    assert all(size(g) <= 300 for g in groups)
    for previous, following in zip(groups, groups[1:]):
        assert following[0] is previous[-1]


def test_long_paragraph_is_not_repeated_as_overlap():
    groups = pack(paragraphs(100, 100, 100, 100), 300, word_count, overlap=60)
    assert [len(g) for g in groups] == [3, 1]


def test_every_paragraph_is_kept_in_order():
    original = paragraphs(120, 30, 200, 10, 90, 250)
    groups = pack(original, 300, word_count, overlap=0)
    assert [p for g in groups for p in g] == original


# --- chunks -------------------------------------------------------------------

@pytest.fixture(scope="module")
def chunks():
    records = [
        lesson_page(1, "Lesson one text.", notes="* A note on lesson one.—T.", lesson=1),
        lesson_page(2, "Lesson two text.", notes="* A note on lesson two.—T.", lesson=2),
    ]
    return build_chunks(records, word_count)


def test_chunks_never_mix_lessons(chunks):
    for chunk in chunks:
        other = "two" if chunk.lesson == 1 else "one"
        assert other not in chunk.text


def test_notes_are_separate_chunks_marked_as_not_charaka(chunks):
    notes = [c for c in chunks if c.kind == "note"]
    texts = [c for c in chunks if c.kind == "text"]
    assert [c.text for c in notes] == ["* A note on lesson one.—T.", "* A note on lesson two.—T."]
    assert all("not part of the Charaka Samhita" in c.attribution for c in notes)
    assert all("translator's note" in c.context for c in notes)
    assert all("note" not in c.text and "not part" not in c.attribution for c in texts)


def test_chunk_ids_are_unique_and_readable(chunks):
    assert [c.chunk_id for c in chunks] == [
        "sutra-01-text-001", "sutra-01-note-001", "sutra-02-text-001", "sutra-02-note-001"]


def test_embedding_text_starts_with_the_lesson(chunks):
    assert embedding_text(chunks[0]).startswith(
        "Charaka Samhita, Sutra Sthana, Lesson I (Dirghanjivitiya)\n\nLesson one text.")


def test_size_counts_the_context_line_too(chunks):
    chunk = chunks[0]
    assert chunk.size == word_count(embedding_text(chunk)) + 2   # + [CLS] and [SEP]


def test_chunk_size_limit_includes_the_context_line():
    records = [lesson_page(1, "\n\n".join(sentence(20) for _ in range(30)))]
    result = build_chunks(records, word_count, max_size=100)
    assert all(c.size <= 100 for c in result)


def test_chunks_round_trip(tmp_path, chunks):
    out = tmp_path / "chunks.jsonl"
    write_chunks(chunks, out)
    assert load_chunks(out) == chunks


# --- the real source (skipped when it has not been downloaded) -------------

needs_source = pytest.mark.skipif(not config.SOURCE_PATH.exists(), reason="source text not downloaded")


@pytest.fixture(scope="module")
def real_lesson_pages():
    return build_structure(ingest_file(config.SOURCE_PATH)).lesson_pages


@needs_source
def test_real_chunks_keep_every_word_and_add_none(real_lesson_pages):
    """Chunks, minus their overlaps, spell out each lesson's text exactly.

    Compared on letters and digits only, so rejoined hyphens and whitespace
    changes do not count, but any lost or invented word does.
    """
    chunks = build_chunks(real_lesson_pages, word_count, max_size=300)
    letters = lambda text: re.sub(r"\W|_", "", text)

    for kind, field in (("text", "main_text"), ("note", "notes_text")):
        lessons = sorted({(r.sthana, r.lesson) for r in real_lesson_pages})
        for key in lessons:
            original = "".join(letters(getattr(r, field)) for r in real_lesson_pages
                               if (r.sthana, r.lesson) == key)
            covered = 0
            for chunk in (c for c in chunks if (c.sthana, c.lesson, c.kind) == (*key, kind)):
                found = original.find(letters(chunk.text), max(0, covered - len(letters(chunk.text))))
                assert 0 <= found <= covered, f"{chunk.chunk_id} is not the next text"
                covered = max(covered, found + len(letters(chunk.text)))
            assert covered == len(original), f"{key} {kind}: text missing from chunks"


@needs_source
@pytest.mark.skipif(not config.MODELS_DIR.exists(), reason="embedding model not downloaded")
def test_real_chunks_fit_the_embedding_model(real_lesson_pages):
    from src.embedder import Embedder
    embedder = Embedder()
    chunks = build_chunks(real_lesson_pages, embedder.count_tokens)
    for chunk in chunks:
        # what the model really sees: the full encoding with [CLS] and [SEP]
        real = len(embedder._tokenizer.encode(embedding_text(chunk)).ids)
        assert real == chunk.size <= MAX_TOKENS, chunk.chunk_id
