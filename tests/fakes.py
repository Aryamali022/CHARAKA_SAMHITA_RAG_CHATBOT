"""Stand-ins for the search system and the answer model, shared by the tests.

They make the tests fast, free and repeatable: no embedding model, no
vector database and no calls to the NVIDIA API.
"""
from src.chunk import Chunk
from src.retrieve import Result


def chunk(n, kind="text", text="Passage text."):
    return Chunk(
        chunk_id=f"sutra-07-{kind}-{n:03d}", source_id="test", kind=kind,
        attribution="Translator's note, not part of the Charaka Samhita" if kind == "note"
        else "Charaka Samhita", sthana="sutra", lesson=7, lesson_title="Navegandharaniya",
        scan_pages=[80, 81], printed_pages=["73", "74"], context="Charaka Samhita, test",
        text=text, word_count=2, size=0,
    )


RESULTS = [
    Result(chunk(1, text="One should not suppress the urge to sneeze."), 0.03, 1, 1),
    Result(chunk(2, "note", text="'Kshavathu' is sneezing.—T."), 0.02, 2, None),
    Result(chunk(3, text="Suppressed urine causes pain in the bladder."), 0.01, None, 2),
]


class FakeRetriever:
    """Returns RESULTS (filtered by kind) and records each query."""

    def __init__(self, results=RESULTS):
        self.results = results
        self.chunks = [r.chunk for r in results]
        self.queries = []
        self.closed = False

    def search(self, query, k=5, method="hybrid", kinds=("text", "note")):
        self.queries.append(query)
        return [r for r in self.results if r.chunk.kind in kinds][:k]

    def close(self):
        self.closed = True


class FakeLLM:
    """Returns canned replies in order and records what it was asked."""
    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, max_tokens=4096, temperature=1.0):
        self.calls.append((system, user))
        return self.replies.pop(0)
