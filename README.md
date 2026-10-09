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

## Setup (Windows / PowerShell)

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Copy-Item .env.example .env   # then add your API key
    python -m pytest

## Status

Phase 1 — source verification and page ingestion.
