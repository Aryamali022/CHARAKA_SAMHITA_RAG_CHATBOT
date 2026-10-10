"""Phase 5: answer a question from the retrieved passages, with citations.

Steps
1. (optional) Rewrite the question into the translation's vocabulary, so a
   question in everyday words finds passages written in 1890s English.
2. Search (Phase 4) for the best passages.
3. Number them: S1, S2, ... for Charaka's text and N1, N2, ... for the
   translator's notes, and ask the model to answer only from them, citing
   the labels.
4. Check the answer in code:
   - citations to labels that were not given are removed and reported;
   - "NOT_IN_TEXT" (or no passages at all) becomes a standard "the text does
     not cover this" reply;
   - if any translator's note is cited, a note saying so is always added,
     whatever the model wrote;
   - every reply ends with the not-medical-advice disclaimer.
Rules that matter are enforced here, not left to the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.llm import LLMError
from src.retrieve import Result, citation

TOP_K = 6   # passages given to the model (about 3,000 tokens)

NOT_IN_TEXT = "NOT_IN_TEXT"

SYSTEM_PROMPT = f"""You answer questions about the Charaka Samhita using only the numbered passages you are given, from the English translation by A. C. Kaviratna (Calcutta, 1890 onward).

Rules:
1. Use only the passages. Do not add facts from your own knowledge, even if you know them.
2. Cite every statement with the label of the passage it comes from, written exactly like [S1] or [S2][N1]: square brackets around the label and nothing else. Use only labels that appear in the passages.
3. Passages labelled S are Charaka's text. Passages labelled N are the translator's notes, not Charaka's words: when you use one, say so, e.g. "According to the translator's note [N1], ...".
4. If the passages do not contain the answer, reply with exactly: {NOT_IN_TEXT}
   This includes questions about modern medicine, drugs, technology or procedures (e.g. chemotherapy, vaccines, insulin): reply {NOT_IN_TEXT} even if the passages discuss a related disease. Do not present the text's remedies as equivalents of modern ones.
5. You are an educational assistant for a historical text. Explain what the text says; do not diagnose the reader, and do not recommend treatments or doses for the reader's own use.
6. Write in plain English: a short paragraph or a few bullet points. Keep the text's own Sanskrit terms where they help.

Reasoning: medium"""

REWRITE_PROMPT = """You write search keywords for finding the answer to a question in an old (1890s) English translation of the Charaka Samhita.

Give only words specific to THIS question: its key ideas, the older English words the translation would use for them, and the Ayurvedic Sanskrit terms for them. Do not add general words such as Charaka, Samhita, Ayurveda, translation, text or treatment-in-general, and do not add terms that are unrelated to the question.

Example
Question: Why does my chest burn after eating spicy food?
Keywords: burning chest heart-burn acidity pungent hot food bile Pitta Amlapitta indigestion

Reply with at most 15 keywords on one line, nothing else.

Reasoning: low"""

NOT_COVERED_REPLY = (
    "The Charaka Samhita does not answer this, at least not in the passages of this "
    "translation that the search found. This translation covers the Sutra, Nidana, Vimana, "
    "Sharira and Indriya Sthanas and Chikitsa Sthana lessons I-XXII; Kalpa and Siddhi "
    "Sthana were never translated. Try rephrasing the question, or using the text's own terms."
)
NOTES_WARNING = ("Note: parts of this answer come from the translator's notes "
                 "(A. C. Kaviratna), not from the Charaka Samhita itself.")
DISCLAIMER = ("Educational use only, not medical advice. It does not diagnose or recommend "
              "treatment. Consult a qualified practitioner.")
UNCITED_WARNING = ("Warning: the model cited no passage, so this answer cannot be checked "
                   "against the text. Treat it with caution.")

# A citation group: "[S1]", "[S1, N2]", and the other forms gpt-oss writes
# (seen in the answer evaluation): "【S1】", "【S1†L3-L5】", "(S3)", "[{S5}]".
_LABEL_ITEM = r"\{?[SN]\d+\}?(?:†[^,;\]】］)]*)?"
CITATION_GROUP_RE = re.compile(rf"[\[【［(]\s*({_LABEL_ITEM}(?:\s*[,;]\s*{_LABEL_ITEM})*)\s*[\]】］)]")
BARE_LABEL_RE = re.compile(r"(?<![\[\w])([SN]\d+)(?![\]\w])")    # "as S2 explains"
INVISIBLE_RE = re.compile("[​‌‍⁠﻿]")      # zero-width characters


@dataclass
class Source:
    label: str          # "S1", "N2", ...
    result: Result

    @property
    def is_note(self) -> bool:
        return self.label.startswith("N")


@dataclass
class Answer:
    question: str
    found: bool                     # False: the text does not cover the question
    text: str                       # the model's answer with [S1]-style citations
    sources: list[Source]           # passages the answer cites, in search-rank order
    passages: list[Source] = field(default_factory=list)   # every passage given to the model
    unknown_labels: list[str] = field(default_factory=list)  # cited but never given; removed
    search_query: str = ""

    @property
    def uses_notes(self) -> bool:
        return any(s.is_note for s in self.sources)


def label_passages(results: list[Result]) -> list[Source]:
    """S1, S2, ... for Charaka's text; N1, N2, ... for translator's notes."""
    counts = {"S": 0, "N": 0}
    sources = []
    for result in results:
        prefix = "N" if result.chunk.kind == "note" else "S"
        counts[prefix] += 1
        sources.append(Source(f"{prefix}{counts[prefix]}", result))
    return sources


def build_user_message(question: str, passages: list[Source]) -> str:
    blocks = []
    for source in passages:
        kind = "translator's note, NOT Charaka's words" if source.is_note else "Charaka's text"
        blocks.append(f"[{source.label}] {citation(source.result.chunk)} ({kind})\n"
                      f"{source.result.chunk.text}")
    return f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(blocks)


def check_citations(text: str, passages: list[Source]) -> tuple[str, list[Source], list[str]]:
    """Normalise every citation form to "[S1]"; drop labels that were not given.

    Bracketed forms ("[S1, S2]", "【S1†L3】", "(S3)", "[{S5}]") are checked
    against the given labels; unknown ones are removed and reported. A bare
    "S2" in a sentence counts only if S2 was given (otherwise it is left as
    ordinary text).

    Returns the cleaned text, the cited sources (in the order given) and the
    unknown labels that were removed.
    """
    given = {s.label: s for s in passages}
    cited: set[str] = set()
    unknown: list[str] = []
    text = INVISIBLE_RE.sub("", text)

    def replace(match: re.Match) -> str:
        kept = []
        for label in re.findall(r"[SN]\d+", match.group(1)):
            if label in given:
                cited.add(label)
                kept.append(f"[{label}]")
            elif label not in unknown:
                unknown.append(label)
        return "".join(kept)

    def replace_bare(match: re.Match) -> str:
        label = match.group(1)
        if label not in given:
            return label
        cited.add(label)
        return f"[{label}]"

    cleaned = CITATION_GROUP_RE.sub(replace, text)
    cleaned = BARE_LABEL_RE.sub(replace_bare, cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)    # space left before punctuation
    return cleaned.strip(), [s for s in passages if s.label in cited], unknown


def rewrite_query(llm, question: str) -> str:
    """The question plus the translation's likely vocabulary, for searching.

    Rewriting only helps the search, so if it fails the question is searched as asked.
    """
    try:
        extra = llm.complete(REWRITE_PROMPT, question, max_tokens=2048)
    except LLMError:
        return question
    return f"{question} {' '.join(extra.replace(',', ' ').split())}"


class Answerer:
    def __init__(self, retriever, llm, k: int = TOP_K, rewrite: bool = False):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.rewrite = rewrite

    def ask(self, question: str, kinds: tuple[str, ...] = ("text", "note")) -> Answer:
        """kinds=("text",) answers from Charaka's text only, without translator's notes."""
        question = " ".join(question.split())
        query = rewrite_query(self.llm, question) if self.rewrite else question
        passages = label_passages(self.retriever.search(query, k=self.k, kinds=kinds))
        if not passages:
            return Answer(question, False, NOT_COVERED_REPLY, [], [], search_query=query)

        reply = self.llm.complete(SYSTEM_PROMPT, build_user_message(question, passages))
        if NOT_IN_TEXT in reply:
            return Answer(question, False, NOT_COVERED_REPLY, [], passages, search_query=query)
        text, sources, unknown = check_citations(reply, passages)
        return Answer(question, True, text, sources, passages, unknown, query)


def format_answer(answer: Answer) -> str:
    lines = [answer.text]
    if answer.found and not answer.sources:
        lines.append(UNCITED_WARNING)
    if answer.uses_notes:
        lines.append(NOTES_WARNING)
    if answer.sources:
        lines.append("Sources:")
        for source in answer.sources:
            kind = ("Translator's note, not Charaka's words" if source.is_note
                    else "Charaka Samhita, tr. A. C. Kaviratna")
            lines[-1] += f"\n  [{source.label}] {citation(source.result.chunk)} ({kind})"
    lines.append(DISCLAIMER)
    return "\n\n".join(lines)
