import numpy as np
import pytest

from src import config
from src.chunk import Chunk, embedding_text
from src.index import COLLECTION, StaleIndexError, open_index, save_index
from src.retrieve import Retriever, citation, format_result

WORDS = ["fever", "cough", "youth", "honey", "longevity", "thirst"]


class FakeEmbedder:
    """Counts a few known words: enough to make 'meaning' search testable offline."""
    model_name = "fake-model"

    def _vector(self, text):
        text = text.lower()
        vector = np.array([text.count(w) for w in WORDS], dtype=np.float32) + 1e-3
        return vector / np.linalg.norm(vector)

    def embed_passages(self, texts):
        return np.array([self._vector(t) for t in texts])

    def embed_query(self, question):
        if "live long" in question:          # a paraphrase that shares no words
            question = "longevity"
        return self._vector(question)


def chunk(n, text, kind="text", printed=("12",), scan=(20,)):
    return Chunk(
        chunk_id=f"sutra-01-{kind}-{n:03d}", source_id="test", kind=kind,
        attribution="Translator's note, not part of the Charaka Samhita" if kind == "note"
        else "Charaka Samhita", sthana="sutra", lesson=1, lesson_title="Dirghanjivitiya",
        scan_pages=list(scan), printed_pages=list(printed), context="Charaka Samhita, test",
        text=text, word_count=len(text.split()), size=0,
    )


CHUNKS = [
    chunk(1, "Fever is treated by fasting and by light gruel."),
    chunk(2, "The Rasayana of Amalaka restores youth and longevity."),
    chunk(3, "Honey cures cough when licked with ghee."),
    chunk(4, "Rasayana means rejuvenation in the original.", kind="note"),
]


def build_index(directory, chunks=CHUNKS, model="fake-model"):
    vectors = FakeEmbedder().embed_passages([embedding_text(c) for c in chunks])
    save_index(vectors, chunks, model, directory)
    return vectors


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    """A real Qdrant database in a temporary folder, filled with the fake vectors."""
    directory = tmp_path_factory.mktemp("index")
    build_index(directory)
    result = Retriever(CHUNKS, open_index(CHUNKS, "fake-model", directory), FakeEmbedder())
    yield result
    result.close()


def ids(results):
    return [r.chunk.chunk_id for r in results]


# --- search methods -----------------------------------------------------------

def test_keyword_search_finds_an_exact_sanskrit_term(retriever):
    results = retriever.search("What is Rasayana?", k=2, method="bm25")
    assert set(ids(results)) == {"sutra-01-text-002", "sutra-01-note-004"}


def test_meaning_search_finds_a_paraphrase_with_no_shared_words(retriever):
    assert retriever.search("how to live long", k=1, method="dense")[0].chunk.chunk_id == "sutra-01-text-002"
    assert retriever.search("how to live long", k=1, method="bm25") == []   # keyword search finds nothing


def test_hybrid_ranks_a_chunk_found_by_both_methods_first(retriever):
    [best] = retriever.search("Rasayana for youth", k=1)
    assert best.chunk.chunk_id == "sutra-01-text-002"
    assert best.dense_rank == 1 and best.bm25_rank is not None
    assert best.score == pytest.approx(1 / 61 + 1 / (60 + best.bm25_rank))


def test_notes_can_be_left_out(retriever):
    results = retriever.search("Rasayana", k=4, kinds=("text",))
    assert all(r.chunk.kind == "text" for r in results)


def test_unknown_method_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("fever", method="magic")


# --- showing results ----------------------------------------------------------

def test_translator_note_is_labelled_in_the_output(retriever):
    note = next(r for r in retriever.search("Rasayana", k=4) if r.chunk.kind == "note")
    shown = format_result(1, note)
    assert "TRANSLATOR'S NOTE - not Charaka's words" in shown
    assert "not part of the Charaka Samhita" in shown


def test_citation_uses_printed_pages_when_they_are_in_order():
    assert citation(chunk(1, "x", printed=("85", "86"), scan=(97, 98))) == \
        "Sutra Sthana, Lesson I (Dirghanjivitiya), pp. 85-86"
    assert citation(chunk(1, "x", printed=("85",), scan=(97,))).endswith(", p. 85")


def test_citation_falls_back_to_scan_pages():
    assert citation(chunk(1, "x", printed=("129", "1016"), scan=(1297, 1298))).endswith(
        ", scan pp. 1297-1298")
    assert citation(chunk(1, "x", printed=(), scan=(11,))).endswith(", scan p. 11")


# --- the vector database ----------------------------------------------------

def test_each_chunk_is_stored_as_a_point_with_its_vector_and_fields(tmp_path):
    vectors = build_index(tmp_path)
    store = open_index(CHUNKS, "fake-model", tmp_path)
    try:
        assert store.count(COLLECTION).count == len(CHUNKS)
        [point] = store.retrieve(COLLECTION, ids=[3], with_vectors=True)
        assert np.allclose(point.vector, vectors[3], atol=1e-6)
        assert point.payload["chunk_id"] == "sutra-01-note-004"
        assert point.payload["kind"] == "note"
        assert point.payload["text"] == CHUNKS[3].text
    finally:
        store.close()


def test_rebuilding_replaces_the_old_collection(tmp_path):
    build_index(tmp_path, chunks=CHUNKS[:2])
    build_index(tmp_path)
    store = open_index(CHUNKS, "fake-model", tmp_path)
    try:
        assert store.count(COLLECTION).count == len(CHUNKS)
    finally:
        store.close()


def test_index_built_from_other_chunks_is_refused(tmp_path):
    build_index(tmp_path)
    changed = CHUNKS[:-1] + [chunk(4, "Edited note text.", kind="note")]
    with pytest.raises(StaleIndexError, match="Chunks changed"):
        open_index(changed, "fake-model", tmp_path)
    with pytest.raises(StaleIndexError, match="built with"):
        open_index(CHUNKS, "another-model", tmp_path)
    open_index(CHUNKS, "fake-model", tmp_path).close()   # a refusal leaves the database usable


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(StaleIndexError, match="python -m src.index"):
        open_index(CHUNKS, "fake-model", tmp_path / "nothing-here")
