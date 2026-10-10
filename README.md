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
    docker compose up -d qdrant       # start the vector database server (Phase 6)
    python -m src.index               # -> Qdrant collection (~5-10 min on CPU)
    python -m src.retrieve "What urges should not be suppressed?"
    python -m scripts.eval_retrieval  # search quality on data/eval/retrieval_questions.jsonl
    python -m src.chat                # ask questions (needs NVIDIA_API_KEY in .env)
    python -m scripts.eval_answers    # answer quality, incl. questions the book cannot answer
    python -m src.api                 # backend server: http://localhost:8000/docs

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
  `storage/models/`), and stored in a **Qdrant** vector database
  (`storage/index/qdrant/`, collection `charaka_chunks`). Qdrant runs embedded
  in Python, so there is no server to install. Each chunk is one point: its
  vector plus its fields (lesson, pages, kind, text) as payload, so the
  database can filter, e.g. Charaka's text only. A question is embedded the
  same way and Qdrant returns the most similar chunks (cosine similarity).
- *Keyword-based:* BM25 (`src/bm25.py`) finds exact terms such as "Rasayana".
  Spellings are normalised so "Çarira", "Sharira" and "sarira" match.
- The two rankings are merged with Reciprocal Rank Fusion.

The collection's metadata records the model and a fingerprint of the chunks;
`src/index.py` refuses an index built from different chunks or another model,
so re-chunking without re-indexing fails loudly. Every result shows its
citation and whether it is Charaka's text or a translator's note.
`scripts/eval_retrieval.py` measures hit@1, hit@5 and MRR for each method on
52 questions whose answering lessons were checked by hand.

**Answers (Phase 5).** `src/chat.py` answers with `openai/gpt-oss-20b` through
NVIDIA's OpenAI-compatible API (`src/llm.py`; key in `.env` as
`NVIDIA_API_KEY`, model and endpoint in `src/config.py`). The question and the
retrieved passages are sent to NVIDIA's servers. For each question
(`src/answer.py`):
1. The top 6 passages are labelled S1, S2, ... (Charaka's text) and N1, N2, ...
   (translator's notes) and given to the model, which must answer only from
   them, cite labels, and reply `NOT_IN_TEXT` when they do not answer it.
2. The code then enforces the rules: citations to passages that were not given
   are removed, "not in text" becomes a standard reply, a note is always added
   when a translator's note is cited, and every reply ends with the
   not-medical-advice disclaimer.

`scripts/eval_answers.py` runs the 52 answerable questions plus 10 the book
cannot answer (COVID-19, insulin doses, chemotherapy, ...). Latest run: 51/52
answered, 49/52 citing a passage from a correct lesson (48 in the run; q04's
answer key was then corrected, as its answer was right), 0 uncited answers,
0 invented citations, and 10/10 unanswerable questions refused. The three
misses are everyday-wording questions where search did not find the lesson.

`--rewrite` first asks the model for the translation's likely vocabulary
(e.g. "insanity lunacy madness" for "lose their mind"). It is off by default:
on the 52 questions it raised hit@5 for everyday wording from 70% to 90% but
lowered it slightly for questions in the book's own terms (100% to 98%), and it
doubles the waiting time.

**Backend server (Phase 6).** Two services run as servers:
- *Qdrant* in Docker (`docker-compose.yml`, port 6333, reachable from this
  computer only). Its data lives in the Docker volume `qdrant_data`. With
  `QDRANT_URL` set in `.env`, every program (API, chat, evaluation scripts)
  connects to it, several at once. Without `QDRANT_URL` the embedded
  database is used, one program at a time; the tests always use a temporary
  embedded database. `python -m src.index --from-embedded` copies an existing
  embedded index to the server without embedding again.
- *The API* (`src/api.py`, FastAPI on port 8000) over the chatbot:
  `GET /api/health`, `POST /api/search`, `POST /api/ask`. Interactive docs at
  `/docs`. The answer rules of Phase 5 apply unchanged; the NVIDIA key stays
  on the server. Requests run in parallel; browser pages from
  `http://localhost:5173` (the React dev server, Phase 7) may call it.

**Web app (Phase 7).** A React + TypeScript frontend in `frontend/` (Vite):
- *Ask*: chat with cited answers. Citations such as [S1] are chips that open the
  passage; every source is badged "Charaka's text" or "Translator's note, not
  Charaka's words"; the backend's warnings and disclaimer are shown as sent.
  Options: Charaka's text only, and question rewriting.
- *Search the text*: passages only, through `/api/search`, quick and free.
- A banner explains what to start when the backend is down or not ready.

Development (two servers; Vite forwards `/api` to FastAPI):

    docker compose up -d qdrant
    python -m src.api                 # http://localhost:8000
    cd frontend && npm install && npm run dev   # http://localhost:5173

Production (one server): `cd frontend && npm run build`, then
`python -m src.api` also serves the app at http://localhost:8000.
Frontend tests: `cd frontend && npm test` (Vitest + React Testing Library,
with a fake backend).

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
Phase 4 — embeddings, Qdrant vector database and hybrid search.
Phase 5 — answers with citations (gpt-oss-20b via NVIDIA).
Phase 6 — backend server: Qdrant in Docker and a FastAPI web API.
Phase 7 — React web app (Ask and Search), served by the API in production.
