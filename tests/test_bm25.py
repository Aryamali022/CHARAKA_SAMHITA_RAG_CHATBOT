from src.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_drops_stopwords_and_punctuation():
    assert tokenize("The Treatment of Fever, thus said Atreya.") == ["treatment", "fever", "atreya"]


def test_translator_and_reader_spellings_match():
    # the translator writes Ç for "sh" and uses diacritics; readers type plain letters
    assert tokenize("Çarira") == tokenize("Sharira") == tokenize("sarira")
    assert tokenize("Vimānam") == tokenize("vimanam")
    assert tokenize("Rasāyana") == tokenize("RASAYANA")


def test_rare_matching_term_ranks_first():
    docs = [tokenize(t) for t in (
        "fever is treated with fasting",
        "rasayana restores youth; fever is mentioned",
        "the diet for fever and for thirst",
    )]
    scores = BM25(docs).scores(tokenize("rasayana fever"))
    assert scores.argmax() == 1


def test_no_matching_terms_score_zero():
    bm25 = BM25([tokenize("fever"), tokenize("cough")])
    assert bm25.scores(tokenize("unknown words")).tolist() == [0.0, 0.0]
    assert bm25.scores([]).tolist() == [0.0, 0.0]


def test_shorter_passage_wins_on_equal_matches():
    bm25 = BM25([tokenize("cough"), tokenize("cough " + "filler " * 30)])
    scores = bm25.scores(tokenize("cough"))
    assert scores[0] > scores[1] > 0
