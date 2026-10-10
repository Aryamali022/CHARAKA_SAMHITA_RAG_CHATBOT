"""Measure how well each search method finds the right lesson.

Questions: data/eval/retrieval_questions.jsonl. Each has the lesson(s) that
answer it, checked against the text by hand. "direct" questions use the
book's own terms; "paraphrase" questions use everyday wording.

Metrics (a result counts as correct when its chunk is from a listed lesson):
  hit@1  share of questions whose first result is correct
  hit@5  share of questions with a correct result in the top 5
  MRR    mean of 1/rank of the first correct result in the top 10 (0 if none)

Run:  python -m scripts.eval_retrieval [--misses]
"""
from __future__ import annotations

import argparse
import json

from src import config
from src.retrieve import METHODS, Retriever, citation

DEPTH = 10
WORKERS = 8     # parallel calls to the answer model when rewriting


def load_questions(path=config.EVAL_QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_correct_rank(results, lessons) -> int | None:
    wanted = {tuple(lesson) for lesson in lessons}
    for rank, result in enumerate(results, 1):
        if (result.chunk.sthana, result.chunk.lesson) in wanted:
            return rank
    return None


def score(ranks: list[int | None]) -> tuple[float, float, float]:
    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks) / n
    hit5 = sum(r is not None and r <= 5 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return hit1, hit5, mrr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--misses", action="store_true", help="list questions hybrid search misses")
    parser.add_argument("--rewrite", action="store_true",
                        help="rewrite each question with the answer model before searching (Phase 5)")
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.rewrite:
        from concurrent.futures import ThreadPoolExecutor

        from src.answer import rewrite_query
        from src.llm import LLM
        llm = LLM()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:   # API calls in parallel
            queries = pool.map(lambda q: rewrite_query(llm, q["question"]), questions)
            for number, (q, query) in enumerate(zip(questions, queries), 1):
                q["query"] = query
                print(f"\rRewriting questions: {number}/{len(questions)}", end="", flush=True)
        print("\n")
    retriever = Retriever.load()
    ranks = {m: [] for m in METHODS}
    top = {}
    try:
        for q in questions:
            for method in METHODS:
                results = retriever.search(q.get("query", q["question"]), k=DEPTH, method=method)
                ranks[method].append(first_correct_rank(results, q["lessons"]))
                if method == "hybrid":
                    top[q["id"]] = results[0]
    finally:
        retriever.close()

    styles = sorted({q["style"] for q in questions})
    per_style = ", ".join(f"{sum(q['style'] == s for q in questions)} {s}" for s in styles)
    print(f"{len(questions)} questions ({per_style})\n")
    print(f"{'method':<8} {'questions':<11} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
    for method in METHODS:
        for style in ["all", *styles]:
            picked = [r for r, q in zip(ranks[method], questions) if style in ("all", q["style"])]
            hit1, hit5, mrr = score(picked)
            print(f"{method:<8} {style:<11} {hit1:>6.0%} {hit5:>6.0%} {mrr:>6.2f}")
        print()

    if args.misses:
        print("Questions hybrid search does not answer in its top 5:")
        for rank, q in zip(ranks["hybrid"], questions):
            if rank is None or rank > 5:
                print(f"  {q['id']} {q['question']}\n      first result: {citation(top[q['id']].chunk)}"
                      f" (correct at rank {rank or '>10'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
