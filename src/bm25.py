"""Keyword search with BM25, the standard ranking formula of search engines.

Meaning-based (embedding) search can miss exact Sanskrit terms such as
"Rasayana" or "pitta"; keyword search finds them reliably. src/retrieve.py
combines the two.

Spelling: the translation writes "Ç" for the sound most readers type as "sh"
("Çarira" = "Sharira") and uses diacritics ("Vimānam"). tokenize() maps both
spellings to the same form, for passages and questions alike.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

K1 = 1.5    # how quickly repeated words stop adding to the score
B = 0.75    # how much long passages are penalised

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers him his how i if in into is
it its itself just me more most my no nor not now of off on once only or other our
out over own same she should so some such than that the their them then there these
they this those through to too under until up upon very was we were what when where
which while who whom why will with would you your thus said shall may one
""".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[çÇśŚṣṢ]", "sh", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("sh", "s")   # "sharira", "sarira", "Çarira" all become "sarira"
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[list[str]]):
        self.term_counts = [Counter(doc) for doc in documents]
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if documents else 0.0
        document_frequency = Counter(term for counts in self.term_counts for term in counts)
        n = len(documents)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5))
                    for term, df in document_frequency.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        """BM25 score of every document for the query terms (0 when none match)."""
        result = np.zeros(len(self.term_counts), dtype=np.float32)
        norm = K1 * (1 - B + B * self.lengths / self.average_length) if self.average_length else 0
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([counts.get(term, 0) for counts in self.term_counts], dtype=np.float32)
            result += idf * tf * (K1 + 1) / (tf + norm)
        return result
