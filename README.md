# Charaka Samhita RAG Chatbot

An educational retrieval-augmented chatbot that answers questions from the
Charaka Samhita, cites sthana / lesson / page, and says so when the text
does not contain an answer.

> **Disclaimer:** This is an educational text assistant, not medical advice.
> It does not diagnose or recommend treatment. Consult a qualified practitioner.

## Source

- **Text:** *Charaka-Samhita*, English translation by Avinash Chandra Kaviratna
  (Calcutta, 1890 onward). The translation is incomplete: Kalpa and Siddhi
  Sthana and the later Chikitsa lessons are not included.
- **Transcription:** Source Library, https://sourcelibrary.org/book/69ef217fb3e2d3a0927f3054
  — licensed **CC BY-SA 4.0**.
- **Original scan:** archive.org identifier `BIUSante_47357` (Bibliothèque
  interuniversitaire de santé, Paris).
- The source file is **not** included in this repository. To get it:
  1. Open the Source Library link above and use **Download** to save the
     plain-text (.txt) version.
  2. Save it as `data/raw/charaka-samhita-ocr.txt`.
  3. Run `python -m scripts.verify_source`. It checks the page count (1988) and a
     SHA-256 of the content from `[Page 1]` onward. The download header is
     excluded because it contains the download date.

## Pipeline

    python -m scripts.verify_source   # check the source file
    python -m src.ingest              # -> data/processed/pages.jsonl + quality report
    python -m src.structure           # -> data/processed/lesson_pages.jsonl + structure.json
    python -m src.chunk               # -> data/processed/chunks.jsonl
    python -m src.index               # -> storage/index/ (embeddings, ~5-10 min on CPU)
    python -m src.retrieve "What urges should not be suppressed?"
    python -m scripts.eval_retrieval  # search quality on data/eval/retrieval_questions.jsonl

**Structure (Phase 2).** The translation prints the divisions in this order:
Sutra, Vimana, Sharira, Indriya, Nidana, Chikitsa (lessons I–XXII), 88 lessons
in all. `src/structure.py` assigns each page's text to its sthana and lesson
and splits pages where a lesson starts mid-page. It leaves out the monthly-part
wrapper pages (press opinions, notices, adverts) that the transcription marks
as text. A small, explicit list (`LESSON_START_FIXES`) covers lesson starts the
transcription leaves unmarked or misnumbered. The run fails if any lesson is
missing. Check `data/processed/structure.json` for the page span of each lesson.

**Chunks (Phase 3).** `src/chunk.py` cuts each lesson into passages that fit
the embedding model: the lesson's context line plus the passage is at most 512
tokens of the model's own tokenizer (about 300 words; Sanskrit words with
diacritics split into many tokens, so sizes are counted in tokens, not words).
A chunk never crosses a lesson, so its citation is always one lesson plus its pages.
Pages are read as continuous text: sentences and hyphenated words broken by a
page break are joined. Whole paragraphs are kept together, and a short
paragraph is repeated at the start of the next chunk as overlap. The
translator's footnotes become separate chunks of kind `note`, whose
`attribution` says they are not part of the Charaka Samhita. Answers that use
them must tell the user so.

**Search (Phase 4).** Hybrid retrieval over all chunks:
- *Meaning-based:* every chunk is embedded once with `BAAI/bge-small-en-v1.5`,
  run locally through `fastembed` (free, no API key, downloaded once into
  `storage/models/`). A question is embedded the same way and compared with
  every chunk (cosine similarity).
- *Keyword-based:* BM25 (`src/bm25.py`) finds exact terms such as "Rasayana".
  Spellings are normalised so "Çarira", "Sharira" and "sarira" match.
- The two rankings are merged with Reciprocal Rank Fusion.

`src/index.py` refuses an index built from different chunks or another model,
so re-chunking without re-indexing fails loudly. Every result shows its
citation and whether it is Charaka's text or a translator's note.
`scripts/eval_retrieval.py` measures hit@1, hit@5 and MRR for each method on
52 questions whose answering lessons were checked by hand.

## Setup (Windows / PowerShell)

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Copy-Item .env.example .env   # then add your API key
    python -m pytest

## Status

Phase 1 — source verification and page ingestion.
Phase 2 — sthana / lesson structure.
Phase 3 — chunking.
Phase 4 — embeddings and hybrid search.
