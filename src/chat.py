"""Ask the Charaka Samhita questions from the terminal.

Run:  python -m src.chat                       (interactive; empty line or "quit" to stop)
      python -m src.chat "What is Rasayana?"   (one question)
Options: --passages  also print every passage the model was given
         --rewrite   rewrite questions into the translation's vocabulary first
"""
from __future__ import annotations

import argparse

from src.answer import Answerer, format_answer
from src.index import IndexNotReadyError
from src.llm import LLM, LLMError
from src.retrieve import Retriever, format_result


def show(answerer: Answerer, question: str, passages: bool) -> None:
    try:
        answer = answerer.ask(question)
    except LLMError as error:
        print(f"Sorry, no answer: {error}\n")
        return
    print("\n" + format_answer(answer) + "\n")
    if answer.unknown_labels:
        print(f"(Removed citations to passages that were not given: {', '.join(answer.unknown_labels)})\n")
    if passages:
        print("Passages given to the model:\n")
        for number, source in enumerate(answer.passages, 1):
            print(f"[{source.label}] " + format_result(number, source.result) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Charaka Samhita a question.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--passages", action="store_true", help="print the passages given to the model")
    parser.add_argument("--rewrite", action="store_true", help="rewrite the question before searching")
    args = parser.parse_args(argv)

    try:
        llm = LLM()
        retriever = Retriever.load()
    except (LLMError, IndexNotReadyError) as error:
        print(error)
        return 1

    answerer = Answerer(retriever, llm, rewrite=args.rewrite)
    try:
        if args.question:
            show(answerer, args.question, args.passages)
            return 0
        print("Ask about the Charaka Samhita (empty line or 'quit' to stop).")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in ("", "quit", "exit"):
                break
            show(answerer, question, args.passages)
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
