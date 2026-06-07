"""Tests for the crew orchestrator, including corrective-RAG fallback."""

from __future__ import annotations

from agentic_rag.agents.crew import AgenticRAGCrew
from agentic_rag.agents.models import RouteName
from agentic_rag.ingestion.vectorstore import RetrievedChunk
from agentic_rag.tools.search_tool import WebSearchHit, WebSearchResult


class _Store:
    """In-memory store stub returning preset chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def count(self):
        return len(self._chunks)

    def query(self, question, *, top_k=None):
        return self._chunks


def test_generate_route_answers_from_model(monkeypatch):
    crew = AgenticRAGCrew(store=_Store([]), use_llm=False)
    monkeypatch.setattr(
        "agentic_rag.agents.crew.generate_answer", lambda *a, **k: "A general answer."
    )
    result = crew.answer("What is the capital of France?")
    assert result.route is RouteName.GENERATE
    assert result.answer == "A general answer."


def test_vectorstore_route_answers_with_citations(monkeypatch):
    chunks = [
        RetrievedChunk(text="Paris is the capital of France.", source="geo.pdf", page=2, score=0.9)
    ]
    crew = AgenticRAGCrew(store=_Store(chunks), use_llm=False)
    monkeypatch.setattr(
        "agentic_rag.agents.crew.generate_grounded_answer",
        lambda *a, **k: "Paris [1].",
    )
    result = crew.answer("What does the document say is the capital of France?")
    assert result.route is RouteName.VECTORSTORE
    assert result.citations
    assert result.citations[0].label == "geo.pdf (p.2)"


def test_irrelevant_context_triggers_web_fallback(monkeypatch):
    # Retrieved chunk is unrelated -> grader fails -> web fallback.
    chunks = [
        RetrievedChunk(text="Penguins are birds.", source="animals.pdf", page=1, score=0.4)
    ]
    crew = AgenticRAGCrew(store=_Store(chunks), use_llm=False)

    monkeypatch.setattr(
        "agentic_rag.agents.crew.web_search",
        lambda *a, **k: WebSearchResult(
            hits=[WebSearchHit(title="Boiling point", url="http://x", content="100C")],
            available=True,
        ),
    )
    monkeypatch.setattr(
        "agentic_rag.agents.crew.generate_grounded_answer",
        lambda *a, **k: "Water boils at 100C [1].",
    )
    result = crew.answer("What is the boiling point of water at sea level?")
    assert result.route is RouteName.WEBSEARCH
    assert result.used_fallback is True
    assert result.citations[0].url == "http://x"


def test_empty_question_handled():
    crew = AgenticRAGCrew(store=_Store([]), use_llm=False)
    result = crew.answer("   ")
    assert "provide a question" in result.answer.lower()
