"""Tests for the backend server (src/api.py), with fake search and a fake model."""
import pytest
from fastapi.testclient import TestClient

from src.answer import DISCLAIMER, NOT_COVERED_REPLY, NOT_IN_TEXT, NOTES_WARNING
from src.api import Services, create_app
from src.llm import LLMError
from tests.fakes import FakeLLM, FakeRetriever


def client_with(retriever=None, llm=None, **errors):
    services = Services(retriever=retriever, llm=llm, store="test store", **errors)
    return TestClient(create_app(services))


@pytest.fixture
def retriever():
    return FakeRetriever()


# --- /api/health -------------------------------------------------------------------

def test_health_when_everything_is_ready(retriever):
    with client_with(retriever, FakeLLM()) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["search"] == {"ready": True, "store": "test store", "error": None, "chunks": 3}
    assert body["answers"] == {"ready": True, "model": "fake-model", "error": None}


def test_health_reports_what_is_missing():
    with client_with(index_error="Cannot reach the Qdrant server.",
                     llm_error="NVIDIA_API_KEY is not set.") as client:
        body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["search"]["error"] == "Cannot reach the Qdrant server."
    assert body["answers"]["error"] == "NVIDIA_API_KEY is not set."


# --- /api/search -------------------------------------------------------------------

def test_search_returns_passages_with_citations_and_kind(retriever):
    with client_with(retriever) as client:
        response = client.post("/api/search", json={"query": "sneeze", "k": 2})
    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["citation"] == "Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74"
    assert first["sthana_name"] == "Sutra Sthana" and first["kind"] == "text"
    assert second["kind"] == "note" and "not part of the Charaka Samhita" in second["attribution"]
    assert (first["dense_rank"], first["bm25_rank"]) == (1, 1)


def test_search_can_leave_out_translator_notes(retriever):
    with client_with(retriever) as client:
        results = client.post("/api/search", json={"query": "sneeze", "text_only": True}).json()["results"]
    assert {r["kind"] for r in results} == {"text"}


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "k": 0}, {"query": "x", "method": "magic"}])
def test_search_rejects_bad_requests(retriever, body):
    with client_with(retriever) as client:
        assert client.post("/api/search", json=body).status_code == 422


def test_search_without_index_is_503_with_the_reason():
    with client_with(index_error="No search index. Run: python -m src.index") as client:
        response = client.post("/api/search", json={"query": "sneeze"})
    assert response.status_code == 503
    assert "python -m src.index" in response.json()["detail"]


# --- /api/ask ------------------------------------------------------------------------

def test_ask_returns_the_cited_answer_and_only_cited_sources(retriever):
    with client_with(retriever, FakeLLM("Never suppress a sneeze [S1].")) as client:
        body = client.post("/api/ask", json={"question": "Why not hold back a sneeze?"}).json()
    assert body["found"] is True
    assert body["answer"] == "Never suppress a sneeze [S1]."
    assert [s["label"] for s in body["sources"]] == ["S1"]
    assert body["uses_translator_notes"] is False and body["notes_warning"] is None
    assert body["disclaimer"] == DISCLAIMER


def test_ask_flags_translator_notes(retriever):
    with client_with(retriever, FakeLLM("Kshavathu means sneezing [N1].")) as client:
        body = client.post("/api/ask", json={"question": "What is Kshavathu?"}).json()
    assert body["uses_translator_notes"] is True
    assert body["notes_warning"] == NOTES_WARNING
    assert body["sources"][0]["kind"] == "note"


def test_ask_not_covered(retriever):
    with client_with(retriever, FakeLLM(NOT_IN_TEXT)) as client:
        body = client.post("/api/ask", json={"question": "How is COVID-19 treated?"}).json()
    assert body["found"] is False and body["answer"] == NOT_COVERED_REPLY and body["sources"] == []
    assert body["disclaimer"] == DISCLAIMER


def test_ask_text_only_gives_the_model_no_notes(retriever):
    llm = FakeLLM("Never suppress it [S1].")
    with client_with(retriever, llm) as client:
        client.post("/api/ask", json={"question": "Sneeze?", "text_only": True})
    assert "[N1]" not in llm.calls[0][1]


def test_ask_reports_removed_citations_and_uncited_answers(retriever):
    with client_with(retriever, FakeLLM("Sneezing is natural [S9].")) as client:
        body = client.post("/api/ask", json={"question": "Sneeze?"}).json()
    assert body["removed_citations"] == ["S9"]
    assert body["uncited_warning"] is not None


def test_ask_without_answer_model_is_503(retriever):
    with client_with(retriever, llm_error="NVIDIA_API_KEY is not set.") as client:
        response = client.post("/api/ask", json={"question": "Sneeze?"})
    assert response.status_code == 503 and "NVIDIA_API_KEY" in response.json()["detail"]


def test_model_failure_during_ask_is_503(retriever):
    class FailingLLM(FakeLLM):
        def complete(self, *args, **kwargs):
            raise LLMError("Rate limit reached. Wait a minute and try again.")

    with client_with(retriever, FailingLLM()) as client:
        response = client.post("/api/ask", json={"question": "Sneeze?"})
    assert response.status_code == 503 and "Rate limit" in response.json()["detail"]


def test_browser_page_from_the_react_dev_server_is_allowed(retriever):
    with client_with(retriever) as client:
        response = client.options("/api/search", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
