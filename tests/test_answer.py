import pytest

from src.answer import (
    DISCLAIMER,
    NOT_COVERED_REPLY,
    NOT_IN_TEXT,
    NOTES_WARNING,
    SYSTEM_PROMPT,
    Answerer,
    build_user_message,
    check_citations,
    format_answer,
    label_passages,
)
from src.llm import LLM, LLMError
from tests.fakes import RESULTS, FakeLLM, FakeRetriever


# --- labelling and the prompt -----------------------------------------------

def test_text_and_notes_get_separate_labels():
    assert [s.label for s in label_passages(RESULTS)] == ["S1", "N1", "S2"]


def test_prompt_shows_each_passage_with_label_citation_and_kind():
    message = build_user_message("Why not hold back a sneeze?", label_passages(RESULTS))
    assert message.startswith("Question: Why not hold back a sneeze?")
    assert "[S1] Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74 (Charaka's text)" in message
    assert "[N1] Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74 " \
           "(translator's note, NOT Charaka's words)" in message
    assert "'Kshavathu' is sneezing.—T." in message


def test_system_prompt_states_the_rules():
    for rule in ("Use only the passages", "translator's notes", NOT_IN_TEXT, "do not diagnose"):
        assert rule in SYSTEM_PROMPT


# --- checking citations -------------------------------------------------------

@pytest.mark.parametrize("reply", [
    # every form below was written by gpt-oss-20b in the answer evaluation
    "Do not hold it back [S1, S2].",
    "Do not hold it back [S1][S2].",
    "Do not hold it back 【S1】【S2】.",           # full-width brackets
    "Do not hold it back 【S1†L1-L3】[S2].",      # line markers
    "Do not hold it back [​S1][S2​].",  # invisible zero-width characters
    "Do not hold it back (S1)(S2).",              # round brackets
    "Do not hold it back [{S1}][{S2}].",          # braces
])
def test_citation_forms_are_normalised(reply):
    text, sources, unknown = check_citations(reply, label_passages(RESULTS))
    assert text == "Do not hold it back [S1][S2]."
    assert [s.label for s in sources] == ["S1", "S2"]
    assert unknown == []


def test_citations_to_passages_never_given_are_removed_and_reported():
    text, sources, unknown = check_citations("It hurts [S2][S9]. Also [N4].", label_passages(RESULTS))
    assert text == "It hurts [S2]. Also."
    assert [s.label for s in sources] == ["S2"]
    assert unknown == ["S9", "N4"]


def test_ordinary_brackets_are_left_alone():
    text, sources, _ = check_citations("The urges [as listed] (happiness) matter [S1].",
                                       label_passages(RESULTS))
    assert text == "The urges [as listed] (happiness) matter [S1]."


def test_bare_label_in_a_sentence_counts_only_if_it_was_given():
    text, sources, unknown = check_citations("Charaka explains it in S2; see also S7.",
                                             label_passages(RESULTS))
    assert text == "Charaka explains it in [S2]; see also S7."
    assert [s.label for s in sources] == ["S2"] and unknown == []


# --- the whole question -> answer flow ----------------------------------------

def test_answer_cites_only_the_passages_it_used():
    llm = FakeLLM("Never suppress a sneeze [S1].")
    answer = Answerer(FakeRetriever(), llm).ask("Why not hold back a sneeze?")
    assert answer.found
    assert [s.label for s in answer.sources] == ["S1"]
    assert [s.label for s in answer.passages] == ["S1", "N1", "S2"]
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT and "[N1]" in user


def test_not_in_text_becomes_the_standard_reply():
    answer = Answerer(FakeRetriever(), FakeLLM(NOT_IN_TEXT)).ask("How is COVID-19 treated?")
    assert not answer.found
    assert answer.text == NOT_COVERED_REPLY and answer.sources == []


def test_no_passages_means_no_model_call():
    llm = FakeLLM()
    answer = Answerer(FakeRetriever(results=[]), llm).ask("Anything?")
    assert not answer.found and llm.calls == []


def test_rewrite_searches_with_the_question_plus_book_vocabulary():
    retriever = FakeRetriever()
    llm = FakeLLM("suppressing the urgings of Nature  sneezing\n", "Answer [S1].")
    Answerer(retriever, llm, rewrite=True).ask("Why not hold back a sneeze?")
    assert retriever.queries == ["Why not hold back a sneeze? suppressing the urgings of Nature sneezing"]
    assert "Question: Why not hold back a sneeze?" in llm.calls[1][1]   # the answer uses the original


def test_failed_rewrite_falls_back_to_the_question_as_asked():
    class RewriteFails(FakeLLM):
        def complete(self, system, user, **kwargs):
            if "search keywords" in system:
                raise LLMError("The model returned no answer (it ran out of tokens while reasoning).")
            return super().complete(system, user, **kwargs)

    retriever = FakeRetriever()
    answer = Answerer(retriever, RewriteFails("Answer [S1]."), rewrite=True).ask("Sneeze?")
    assert retriever.queries == ["Sneeze?"] and answer.found


# --- what the user sees ----------------------------------------------------------

def test_translator_note_warning_is_added_by_code_whatever_the_model_says():
    answer = Answerer(FakeRetriever(), FakeLLM("Kshavathu means sneezing [N1].")).ask("Kshavathu?")
    shown = format_answer(answer)
    assert NOTES_WARNING in shown
    assert "[N1] Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74 " \
           "(Translator's note, not Charaka's words)" in shown


def test_no_note_warning_when_only_charaka_is_cited():
    answer = Answerer(FakeRetriever(), FakeLLM("Never suppress it [S1].")).ask("Sneeze?")
    assert NOTES_WARNING not in format_answer(answer)


def test_every_reply_ends_with_the_disclaimer():
    for reply in ("Never suppress it [S1].", NOT_IN_TEXT, "No citations here."):
        answer = Answerer(FakeRetriever(), FakeLLM(reply)).ask("Sneeze?")
        assert format_answer(answer).endswith(DISCLAIMER)


def test_uncited_answer_carries_a_warning():
    answer = Answerer(FakeRetriever(), FakeLLM("Sneezing is natural.")).ask("Sneeze?")
    assert "cited no passage" in format_answer(answer)


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(LLMError, match="NVIDIA_API_KEY is not set"):
        LLM()
