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
- The source file is **not** included in this repository. Download it and place
  it at `data/raw/charaka-samhita-ocr.txt` (verification script: Phase 1).

## Setup (Windows / PowerShell)

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    Copy-Item .env.example .env   # then add your API key
    python -m pytest

## Status

Phase 0 — project structure.
