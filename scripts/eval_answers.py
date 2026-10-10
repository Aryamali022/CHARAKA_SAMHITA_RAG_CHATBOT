"""Measure the chatbot's answers (Phase 5). Calls the answer model once per question.

Answerable questions: data/eval/retrieval_questions.jsonl (with their lessons).
Unanswerable questions: data/eval/unanswerable_questions.jsonl (the book does
not cover them; the right reply is "the text does not cover this").

Reported:
  answered            answerable questions that got an answer
  cites right lesson  answers citing at least one passage from a correct lesson
  uncited             answers with no valid citation
  invented citations  citations to passages that were never given (removed by code)
  refused correctly   unanswerable questions answered with "not covered"
  notes cited         answers citing a translator's note, and how many of those
                      also say "translator" in the model's own words (the code
                      adds the warning either way)

Every answer is saved to data/processed/answer_eval.jsonl for reading by hand.

Run:  python -m scripts.eval_answers [--rewrite] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.answer import Answerer
from src.llm import LLM, LLMError
from src.retrieve import Retriever

WORKERS = 8     # questions answered in parallel (the model API is the slow part)


class OneSearchAtATime:
    """Lets threads share the retriever: searches run one at a time (the local
    Qdrant database and the embedder are not shared between threads), while
    the slow model calls run in parallel."""

    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self._lock = threading.Lock()

    def search(self, *args, **kwargs):
        with self._lock:
            return self._retriever.search(*args, **kwargs)


def load(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(answerer: Answerer, q: dict) -> dict:
    try:
        answer = answerer.ask(q["question"])
    except LLMError as error:
        return {**q, "error": str(error)}
    cited = [(s.result.chunk.sthana, s.result.chunk.lesson) for s in answer.sources]
    return {
        **q, "found": answer.found, "text": answer.text, "search_query": answer.search_query,
        "cited": [s.result.chunk.chunk_id for s in answer.sources],
        "right_lesson": any(list(c) in q.get("lessons", []) for c in cited),
        "notes_cited": answer.uses_notes,
        "says_translator": "translator" in answer.text.lower(),
        "unknown_labels": answer.unknown_labels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rewrite", action="store_true", help="rewrite questions before searching")
    parser.add_argument("--limit", type=int, help="only the first N questions of each set")
    args = parser.parse_args(argv)

    answerable = load(config.EVAL_QUESTIONS_PATH)[:args.limit]
    unanswerable = load(config.UNANSWERABLE_QUESTIONS_PATH)[:args.limit]
    retriever = Retriever.load()
    answerer = Answerer(OneSearchAtATime(retriever), LLM(), rewrite=args.rewrite)
    questions = answerable + unanswerable
    records = []
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for number, record in enumerate(pool.map(lambda q: evaluate(answerer, q), questions), 1):
                records.append(record)
                print(f"\rAnswering {number}/{len(questions)}", end="", flush=True)
    finally:
        retriever.close()
    print("\n")

    config.ANSWER_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.ANSWER_EVAL_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def share(part, whole):
        return f"{len(part)}/{len(whole)} ({len(part) / len(whole):.0%})" if whole else "0/0"

    ok = [r for r in records if "error" not in r]
    a = [r for r in ok if "lessons" in r]
    u = [r for r in ok if "lessons" not in r]
    found = [r for r in a if r["found"]]
    notes = [r for r in ok if r.get("notes_cited")]
    print(f"Answerable questions   : {len(a)}")
    print(f"  answered             : {share(found, a)}")
    print(f"  cites right lesson   : {share([r for r in a if r['right_lesson']], a)}")
    print(f"  uncited answers      : {share([r for r in found if not r['cited']], found)}")
    print(f"  invented citations   : {sum(len(r['unknown_labels']) for r in ok)} (removed by code)")
    print(f"Unanswerable questions : {len(u)}")
    print(f"  refused correctly    : {share([r for r in u if not r['found']], u)}")
    print(f"Answers citing notes   : {len(notes)}, of which the model itself says 'translator': "
          f"{sum(r['says_translator'] for r in notes)}")
    errors = [r for r in records if "error" in r]
    if errors:
        print(f"Errors                 : {len(errors)} ({errors[0]['error']})")
    print(f"\nAll answers saved to {config.ANSWER_EVAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
