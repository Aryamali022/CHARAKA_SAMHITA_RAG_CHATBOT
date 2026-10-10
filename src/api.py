"""Phase 6: the backend server, a web API over the chatbot.

The React app (Phase 7), or any other client, uses the chatbot through these
endpoints instead of importing its Python code:

  GET  /api/health   is everything ready: vector database, chunks, answer model
  POST /api/search   hybrid search only (no answer model): passages with citations
  POST /api/ask      search + cited answer, with every rule of src/answer.py

Interactive documentation: http://localhost:8000/docs

The NVIDIA key stays on the server; clients never see it. Requests run in
parallel threads: searches share one retriever through SerialSearch, and the
slow answer-model calls run side by side.

It also serves the React app (Phase 7) at / once it is built
(cd frontend && npm run build), so one server runs the whole chatbot.

Run:  docker compose up -d qdrant     (the vector database server)
      python -m src.api              (this server, http://localhost:8000)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.answer import DISCLAIMER, NOTES_WARNING, UNCITED_WARNING, Answerer
from src.chunk import Chunk
from src.index import IndexNotReadyError
from src.llm import LLM, LLMError
from src.retrieve import METHODS, Retriever, SerialSearch, citation
from src.structure import STHANAS

STHANA_NAMES = {s.key: s.name for s in STHANAS}
FRONTEND_DIST = config.PROJECT_ROOT / "frontend" / "dist"


# --- what the server holds ------------------------------------------------------

@dataclass
class Services:
    """The loaded chatbot parts. A part that failed to load is None, with the reason."""
    retriever: object | None = None      # Retriever (or a test fake with search/close)
    llm: object | None = None            # LLM (or a test fake with complete)
    index_error: str | None = None
    llm_error: str | None = None
    store: str = ""                      # where the vectors are, for /api/health

    def __post_init__(self):
        self.searcher = SerialSearch(self.retriever) if self.retriever is not None else None


def load_services() -> Services:
    store = f"Qdrant server {config.QDRANT_URL}" if config.QDRANT_URL else "embedded Qdrant database"
    retriever = llm = index_error = llm_error = None
    try:
        retriever = Retriever.load()
    except IndexNotReadyError as error:
        index_error = str(error)
    try:
        llm = LLM()
    except LLMError as error:
        llm_error = str(error)
    return Services(retriever, llm, index_error, llm_error, store)


# --- request and response shapes (shown in /docs) -------------------------------

class PassageOut(BaseModel):
    chunk_id: str
    kind: Literal["text", "note"] = Field(description="'text' = Charaka; 'note' = translator's note")
    attribution: str
    sthana: str
    sthana_name: str
    lesson: int
    lesson_title: str
    citation: str
    scan_pages: list[int]
    printed_pages: list[str]
    text: str


class SearchHit(PassageOut):
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class SourceOut(PassageOut):
    label: str = Field(description="as cited in the answer, e.g. 'S1' or 'N1'")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    method: Literal[METHODS] = "hybrid"
    text_only: bool = Field(False, description="leave out translator's notes")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    rewrite: bool = Field(False, description="rewrite into the translation's vocabulary first (slower)")
    text_only: bool = Field(False, description="answer from Charaka's text only, without translator's notes")


class AskResponse(BaseModel):
    question: str
    found: bool = Field(description="False when the text does not cover the question")
    answer: str = Field(description="the answer with [S1]-style citations, or the 'not covered' reply")
    sources: list[SourceOut] = Field(description="only the passages the answer cites")
    uses_translator_notes: bool
    notes_warning: str | None
    uncited_warning: str | None
    removed_citations: list[str] = Field(description="citations to passages never given; removed")
    disclaimer: str


def passage_fields(chunk: Chunk) -> dict:
    return dict(
        chunk_id=chunk.chunk_id, kind=chunk.kind, attribution=chunk.attribution,
        sthana=chunk.sthana, sthana_name=STHANA_NAMES.get(chunk.sthana, chunk.sthana),
        lesson=chunk.lesson, lesson_title=chunk.lesson_title, citation=citation(chunk),
        scan_pages=chunk.scan_pages, printed_pages=chunk.printed_pages, text=chunk.text,
    )


# --- the app ----------------------------------------------------------------------

def create_app(services: Services | None = None) -> FastAPI:
    """The API. Tests pass their own services; the real server loads them at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services if services is not None else load_services()
        yield
        if services is None and app.state.services.retriever is not None:
            app.state.services.retriever.close()

    app = FastAPI(title="Charaka Samhita RAG Chatbot API", version="1.0", lifespan=lifespan,
                  description="Search and cited answers over the Charaka Samhita "
                              "(tr. A. C. Kaviratna). Educational use only, not medical advice.")
    app.add_middleware(CORSMiddleware, allow_origins=config.API_CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def get_services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/api/health")
    def health(request: Request) -> dict:
        s = get_services(request)
        search_ready = s.retriever is not None
        answers_ready = search_ready and s.llm is not None
        return {
            "status": "ok" if answers_ready else "degraded",
            "search": {"ready": search_ready, "store": s.store, "error": s.index_error,
                       "chunks": len(s.retriever.chunks) if search_ready else 0},
            "answers": {"ready": answers_ready, "model": getattr(s.llm, "model", None),
                        "error": s.llm_error},
        }

    @app.post("/api/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        s = get_services(request)
        if s.searcher is None:
            raise HTTPException(503, s.index_error or "Search is not available.")
        kinds = ("text",) if body.text_only else ("text", "note")
        results = s.searcher.search(body.query, k=body.k, method=body.method, kinds=kinds)
        return SearchResponse(query=body.query, results=[
            SearchHit(**passage_fields(r.chunk), score=r.score, dense_rank=r.dense_rank,
                      bm25_rank=r.bm25_rank) for r in results])

    @app.post("/api/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        s = get_services(request)
        if s.searcher is None:
            raise HTTPException(503, s.index_error or "Search is not available.")
        if s.llm is None:
            raise HTTPException(503, s.llm_error or "The answer model is not available.")
        kinds = ("text",) if body.text_only else ("text", "note")
        try:
            answer = Answerer(s.searcher, s.llm, rewrite=body.rewrite).ask(body.question, kinds=kinds)
        except LLMError as error:
            raise HTTPException(503, str(error)) from error
        return AskResponse(
            question=answer.question, found=answer.found, answer=answer.text,
            sources=[SourceOut(**passage_fields(src.result.chunk), label=src.label)
                     for src in answer.sources],
            uses_translator_notes=answer.uses_notes,
            notes_warning=NOTES_WARNING if answer.uses_notes else None,
            uncited_warning=UNCITED_WARNING if answer.found and not answer.sources else None,
            removed_citations=answer.unknown_labels,
            disclaimer=DISCLAIMER,
        )

    # The built React app (cd frontend && npm run build), served last so the
    # /api routes above take precedence. In development Vite serves it instead.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()   # for: uvicorn src.api:app


def main() -> int:
    import uvicorn
    print(f"API docs: http://{config.API_HOST}:{config.API_PORT}/docs")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
